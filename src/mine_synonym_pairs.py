"""
Mine training pairs for fine-tuning an embedding model — no manual labeling
needed. HEP abstracts constantly define acronyms on first use:

    "...weakly interacting massive particles (WIMPs)..."
    "...using effective field theory (EFT) methods..."

Every one of those is a free, genuinely-semantic synonym pair (unlike the
word-order/morphology variants the TF-IDF cluster backend already handles
well) — exactly the kind of case ("ALP" vs "axion-like particle") the
TF-IDF backend can't merge, which is the whole reason to fine-tune an
embedding model in the first place.

Run:
    python mine_synonym_pairs.py

Reads config.RAW_METADATA_PATH, writes config.SYNONYM_PAIRS_PATH:
    {"phrase_a": "weakly interacting massive particles", "phrase_b": "WIMPs",
     "source": "acronym_definition", "doc_id": "2401.01234v1"}

Method: a simplified Schwartz-Hearst style heuristic. For each "(ACRONYM)"
in the text, scan backward through the preceding words and check whether
their initials — skipping short function words like "of"/"the"/"for" —
spell out the acronym's letters in order. A trailing lowercase "s" on the
acronym (WIMPs, ALPs) is treated as a plural marker, not an extra initial.

If config.CLUSTERS_PATH already exists (i.e. you've run cluster_phrases.py),
this also emits same-cluster phrase pairs as supplementary positives — these
reinforce the easy word-order/morphology cases the current backend already
gets right, which isn't the interesting training signal, but doesn't hurt.

Caveat: this is a heuristic, not a certainty. Skim a sample of the output
before trusting it as training data — a wrong pair teaches the model false
synonymy, and a small amount of noise doesn't come out in the wash the way
it might with a much larger dataset.
"""
import json
import logging
import random
import re
from itertools import combinations

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mine_synonym_pairs")

ACRONYM_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9]{1,9})\)")
STOPWORDS_SKIP = {"of", "the", "a", "an", "for", "in", "on", "and", "to", "with", "at"}
MAX_SAME_CLUSTER_PAIRS_PER_CLUSTER = 6  # cap combinatorial blowup on large clusters


def _acronym_letters(acronym: str):
    # a trailing lowercase 's' is almost always a plural marker (WIMPs, ALPs),
    # not part of the initialism itself
    if len(acronym) > 2 and acronym[-1] == "s" and acronym[:-1].isupper():
        acronym = acronym[:-1]
    return [c for c in acronym if c.isalpha()]


def _find_long_form(preceding_text: str, acronym: str):
    """Scan backward for a word window whose initials spell the acronym."""
    letters = _acronym_letters(acronym)
    words = re.findall(r"[A-Za-z]+", preceding_text)  # hyphens act as word breaks
    if not words or not letters:
        return None
    max_window = min(len(words), len(letters) + 4)
    for window_size in range(len(letters), max_window + 1):
        candidate = words[-window_size:]
        li, wi = len(letters) - 1, len(candidate) - 1
        used = []
        matched = True
        while li >= 0:
            if wi < 0:
                matched = False
                break
            w = candidate[wi]
            if w[0].lower() == letters[li].lower():
                used.append(w)
                li -= 1
                wi -= 1
            elif w.lower() in STOPWORDS_SKIP:
                used.append(w)
                wi -= 1
            else:
                matched = False
                break
        if matched:
            used.reverse()
            return " ".join(used).lower()
    return None


def mine_from_text(text: str):
    pairs = []
    for m in ACRONYM_RE.finditer(text):
        acronym = m.group(1)
        if not any(c.isupper() for c in acronym) or len(acronym) < 2:
            continue
        long_form = _find_long_form(text[:m.start()], acronym)
        if long_form and long_form != acronym.lower():
            pairs.append((long_form, acronym))
    return pairs


def mine_acronym_pairs():
    n_docs = 0
    seen = set()
    pairs = []
    with open(config.RAW_METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_docs += 1
            text = f"{rec['title']}. {rec['abstract']}"
            for long_form, acronym in mine_from_text(text):
                key = (long_form, acronym)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"phrase_a": long_form, "phrase_b": acronym,
                              "source": "acronym_definition", "doc_id": rec["id"]})
    log.info("Scanned %d documents, mined %d unique acronym-definition pairs", n_docs, len(pairs))
    return pairs


def same_cluster_pairs():
    if not config.CLUSTERS_PATH.exists():
        log.info("No phrase_clusters.json found — skipping same-cluster supplementary pairs "
                 "(run cluster_phrases.py first if you want these too).")
        return []
    with open(config.CLUSTERS_PATH, encoding="utf-8") as f:
        clusters = json.load(f)
    pairs = []
    for cid, meta in clusters.items():
        members = meta["members"]
        if len(members) < 2:
            continue
        combos = list(combinations(members, 2))
        if len(combos) > MAX_SAME_CLUSTER_PAIRS_PER_CLUSTER:
            combos = random.sample(combos, MAX_SAME_CLUSTER_PAIRS_PER_CLUSTER)
        for a, b in combos:
            pairs.append({"phrase_a": a, "phrase_b": b, "source": "same_cluster", "doc_id": None})
    log.info("Added %d supplementary same-cluster pairs from %d clusters", len(pairs), len(clusters))
    return pairs


def main():
    random.seed(42)
    pairs = mine_acronym_pairs() + same_cluster_pairs()
    if not pairs:
        log.warning("No pairs mined — check that %s exists and has content.", config.RAW_METADATA_PATH)
        return
    with open(config.SYNONYM_PAIRS_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info("Wrote %d total pairs to %s", len(pairs), config.SYNONYM_PAIRS_PATH)
    log.info("Spot-check a sample before training:")
    for p in random.sample(pairs, min(10, len(pairs))):
        log.info("  %-45s <-> %-20s (%s)", p["phrase_a"], p["phrase_b"], p["source"])


if __name__ == "__main__":
    main()
