"""
Stage 1 — Download HEP article metadata from the arXiv API.

Run:
    python fetch_arxiv.py            # fetch (resumes automatically if interrupted)
    python fetch_arxiv.py --fresh    # ignore any checkpoint/output and start over

Output: one JSON object per line at config.RAW_METADATA_PATH, e.g.
    {"id": "2401.01234v1", "title": "...", "abstract": "...",
     "categories": ["hep-ph", "hep-ex"], "primary_category": "hep-ph",
     "published": "2024-01-02T00:00:00Z", "updated": "...", "authors": [...]}

Robustness notes (see https://info.arxiv.org/help/api/user-manual.html):
  - Deep `start` offsets are a known source of intermittent 500 errors, and
    arXiv's manual itself recommends against single queries that return
    more than ~1000 results ("We recommend to refine queries which return
    more than 1,000 results, or at least request smaller slices"). A whole
    calendar year of hep-ph+hep-th easily exceeds that. So instead of one
    query per year, we recursively bisect the date range: probe how many
    results a range matches, and if it's above config.SAFE_MAX_RESULTS_PER_SLICE,
    split the range in half and recurse. Every actual paginated `start`
    offset then stays small.
  - Transient 500/503/connection errors are retried with exponential
    backoff (config.MAX_RETRIES / RETRY_BACKOFF_*).
  - Progress is checkpointed per date-slice (config.FETCH_CHECKPOINT_PATH).
    If the process crashes or is interrupted, re-running `python
    fetch_arxiv.py` skips slices already completed and appends to the
    existing output file instead of starting over. A per-record id de-dup
    guard also protects against writing the same paper twice if slice
    boundaries happen to shift between runs (see the NOTE in main()).
  - submittedDate ranges use 12-digit GMT timestamps: YYYYMMDDHHMM.
"""
import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Any, Dict, Iterator, Set, Tuple

import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_arxiv")

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"
API_URL = "http://export.arxiv.org/api/query"


class MaxResultsReached(Exception):
    """Internal signal used to unwind all pending recursion once the
    configured MAX_TOTAL_RESULTS cap is hit."""


# ---- Query construction ----------------------------------------------------

def _build_query(d_start: date, d_end: date) -> str:
    cats = " OR ".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    start_ts = d_start.strftime("%Y%m%d") + "0000"
    end_ts = d_end.strftime("%Y%m%d") + "2359"
    return f"({cats}) AND submittedDate:[{start_ts} TO {end_ts}]"


# ---- HTTP + XML parsing, with retries ---------------------------------------

def _request_with_retry(params: dict) -> ET.Element:
    last_exc = None
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            time.sleep(config.REQUEST_DELAY_SECONDS)   # <-- add this
            return root
        except (requests.exceptions.RequestException, ET.ParseError) as exc:
            last_exc = exc
            wait = min(config.RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt),
                       config.RETRY_BACKOFF_MAX_SECONDS)
            log.warning("Request failed (attempt %d/%d): %s — retrying in %.0fs",
                        attempt + 1, config.MAX_RETRIES, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Giving up after {config.MAX_RETRIES} attempts") from last_exc


def _text(entry: ET.Element, tag: str, ns: str = ATOM_NS) -> str:
    el = entry.find(f"{ns}{tag}")
    return " ".join(el.text.split()) if (el is not None and el.text) else ""


def _parse_entry(entry: ET.Element) -> Dict[str, Any]:
    arxiv_id = _text(entry, "id").split("/abs/")[-1]
    categories = [c.get("term") for c in entry.findall(f"{ATOM_NS}category")]
    primary_el = entry.find(f"{ARXIV_NS}primary_category")
    primary_category = primary_el.get("term") if primary_el is not None else (categories[0] if categories else "")
    authors = [
        a.find(f"{ATOM_NS}name").text.strip()
        for a in entry.findall(f"{ATOM_NS}author")
        if a.find(f"{ATOM_NS}name") is not None and a.find(f"{ATOM_NS}name").text
    ]
    return {
        "id": arxiv_id,
        "title": _text(entry, "title"),
        "abstract": _text(entry, "summary"),
        "categories": categories,
        "primary_category": primary_category,
        "published": _text(entry, "published"),
        "updated": _text(entry, "updated"),
        "authors": authors,
    }


def _count_matches(query: str) -> int:
    """Cheap probe (max_results=1) purely to read <opensearch:totalResults>."""
    root = _request_with_retry({"search_query": query, "start": 0, "max_results": 1})
    el = root.find(f"{OPENSEARCH_NS}totalResults")
    return int(el.text) if el is not None and el.text else 0


def _paginate(query: str, total: int, counter: Dict[str, int]) -> Iterator[Dict[str, Any]]:
    start = 0
    page_size = config.MAX_RESULTS_PER_REQUEST
    while start < total:
        root = _request_with_retry({
            "search_query": query, "start": start, "max_results": page_size,
            "sortBy": "submittedDate", "sortOrder": "ascending",
        })
        entries = root.findall(f"{ATOM_NS}entry")
        if not entries:
            break
        for entry in entries:
            yield _parse_entry(entry)
            counter["n"] += 1
            if config.MAX_TOTAL_RESULTS and counter["n"] >= config.MAX_TOTAL_RESULTS:
                raise MaxResultsReached()
        start += len(entries)
        if start < total:
            time.sleep(config.REQUEST_DELAY_SECONDS)


# ---- Adaptive date-range bisection ------------------------------------------

def _fetch_range(d_start: date, d_end: date, done_ranges: Set[Tuple[str, str]],
                  counter: Dict[str, int]) -> Iterator[Dict[str, Any]]:
    key = (d_start.isoformat(), d_end.isoformat())
    if key in done_ranges:
        log.info("Skipping already-completed range %s to %s (from checkpoint)", *key)
        return

    query = _build_query(d_start, d_end)
    total = _count_matches(query)
    if total == 0:
        done_ranges.add(key)
        return

    if total > config.SAFE_MAX_RESULTS_PER_SLICE and d_end > d_start:
        mid = d_start + (d_end - d_start) // 2
        log.info("Range %s..%s has %d matches (> %d) — splitting at %s",
                  d_start, d_end, total, config.SAFE_MAX_RESULTS_PER_SLICE, mid)
        yield from _fetch_range(d_start, mid, done_ranges, counter)
        yield from _fetch_range(mid + timedelta(days=1), d_end, done_ranges, counter)
        return

    if total > config.SAFE_MAX_RESULTS_PER_SLICE:
        log.warning("Single day %s has %d matches (> %d) and can't be split further; "
                     "paginating anyway — deep offsets here may be flaky.",
                     d_start, total, config.SAFE_MAX_RESULTS_PER_SLICE)

    log.info("Fetching range %s..%s (%d matches)", d_start, d_end, total)
    yield from _paginate(query, total, counter)
    done_ranges.add(key)


# ---- Checkpointing -----------------------------------------------------------

def _load_checkpoint() -> Set[Tuple[str, str]]:
    if not config.FETCH_CHECKPOINT_PATH.exists():
        return set()
    with open(config.FETCH_CHECKPOINT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {tuple(pair) for pair in data.get("completed_ranges", [])}


def _save_checkpoint(done_ranges: Set[Tuple[str, str]]) -> None:
    with open(config.FETCH_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"completed_ranges": sorted(done_ranges)}, f)


def _load_existing_ids() -> Set[str]:
    if not config.RAW_METADATA_PATH.exists():
        return set()
    ids = set()
    with open(config.RAW_METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


# ---- Entry point --------------------------------------------------------------

def main(fresh: bool = False):
    if fresh:
        config.FETCH_CHECKPOINT_PATH.unlink(missing_ok=True)
        config.RAW_METADATA_PATH.unlink(missing_ok=True)

    done_ranges = _load_checkpoint()
    seen_ids = _load_existing_ids()
    resuming = bool(done_ranges or seen_ids)
    if resuming:
        log.info("Resuming: %d completed range(s) checkpointed, %d record(s) already saved",
                  len(done_ranges), len(seen_ids))

    d_start = date.fromisoformat(config.START_DATE)
    d_end = date.fromisoformat(config.END_DATE) - timedelta(days=1)  # END_DATE is exclusive

    counter = {"n": len(seen_ids)}
    n_new = 0
    mode = "a" if resuming else "w"
    try:
        with open(config.RAW_METADATA_PATH, mode, encoding="utf-8") as f:
            for record in _fetch_range(d_start, d_end, done_ranges, counter):
                # NOTE: the checkpoint above skips whole date-slices for speed,
                # but if config (dates/categories/SAFE_MAX_RESULTS_PER_SLICE)
                # changes between a crash and a resume, bisection boundaries
                # can shift slightly. This id check is what actually
                # guarantees no duplicate rows end up in the output file.
                if record["id"] in seen_ids:
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                seen_ids.add(record["id"])
                n_new += 1
                if n_new % 500 == 0:
                    _save_checkpoint(done_ranges)
                    log.info("...%d new records so far", n_new)
    except MaxResultsReached:
        log.info("Reached MAX_TOTAL_RESULTS=%d, stopping.", config.MAX_TOTAL_RESULTS)
    finally:
        _save_checkpoint(done_ranges)

    log.info("Done. %d new record(s) added this run, %d total in %s",
              n_new, len(seen_ids), config.RAW_METADATA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing checkpoint/output and start over.")
    args = parser.parse_args()
    main(fresh=args.fresh)
