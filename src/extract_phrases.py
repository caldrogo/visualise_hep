"""
Stage 2 — Extract candidate topic phrases from titles + abstracts.

Setup (once):
    pip install spacy
    python -m spacy download en_core_web_sm
    # Optional, better on physics jargon: pip install scispacy + a sci model,
    # then set SPACY_MODEL in config.py to that model's package name.

Run:
    python extract_phrases.py

Reads config.RAW_METADATA_PATH, writes one JSON line per document to
config.PHRASES_PATH:
    {"id": "2401.01234v1", "year": 2024, "phrases": ["dark matter direct detection", ...]}

Method: spaCy noun-chunk extraction over "title. abstract", restricted to
2-5 token spans (config.PHRASE_MIN/MAX_TOKENS), trimmed of leading
determiners/pronouns/prepositions, lemmatized token-by-token (while
preserving short ALL-CAPS acronyms like QCD/LHC/CP/CKM), and filtered
against a stopword + generic-word list (config.EXTRA_STOPWORDS). This is a
light, fast keyphrase proxy — good enough to feed the clustering stage,
which is what does the heavy lifting of merging near-duplicate phrasings.
"""
import json
import re
import logging
from typing import List

import spacy

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_phrases")

MATH_INLINE_RE = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip inline LaTeX math (so '$m_\\chi$' doesn't pollute chunks) and collapse whitespace."""
    text = MATH_INLINE_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _normalize_token(tok) -> str:
    if tok.text.isupper() and 2 <= len(tok.text) <= 6:
        return tok.text  # preserve acronyms: QCD, LHC, CP, CKM, BSM, ...
    return tok.lemma_.lower()


def _trim_leading(tokens: list) -> list:
    """Drop leading determiners/pronouns/prepositions/numbers from a noun chunk."""
    while tokens and tokens[0].pos_ in ("DET", "PRON", "ADP", "NUM"):
        tokens = tokens[1:]
    return tokens


def _is_valid_phrase(tokens: list, phrase_text: str) -> bool:
    n = len(tokens)
    if not (config.PHRASE_MIN_TOKENS <= n <= config.PHRASE_MAX_TOKENS):
        return False
    if all(t.is_stop for t in tokens):
        return False
    if not any(c.isalpha() for c in phrase_text):
        return False
    words = phrase_text.split()
    if words[-1] in config.EXTRA_STOPWORDS:
        return False
    if tokens[-1].pos_ not in ("NOUN", "PROPN"):  # phrase should be headed by a noun
        return False
    return True


def extract_phrases_from_doc(doc) -> List[str]:
    phrases = set()
    for chunk in doc.noun_chunks:
        tokens = _trim_leading(list(chunk))
        if not tokens:
            continue
        phrase_text = " ".join(_normalize_token(t) for t in tokens)
        if _is_valid_phrase(tokens, phrase_text):
            phrases.add(phrase_text)
    return sorted(phrases)

def filter_primary_category(records: List[dict]) -> List[dict]:
    """Keep only records whose *primary* arXiv category is in config.ARXIV_CATEGORIES.
 
    fetch_arxiv.py's search_query matches a paper if ANY of its listed
    categories — primary or cross-listed — is in that set, so raw_metadata.jsonl
    can include papers that are only tangentially HEP (e.g. primary category
    astro-ph.CO, cross-listed under hep-ph). This narrows the corpus down to
    papers that are actually about HEP first, which matters here: a paper
    whose main subject is cosmology shouldn't count toward "HEP topic X is
    trending" just because it touches hep-ph in passing.
    """
    kept = [r for r in records if r.get("primary_category") in config.ARXIV_CATEGORIES]
    dropped = len(records) - len(kept)
    if dropped:
        log.info("Dropped %d/%d records whose primary_category is outside %s "
                  "(cross-listed, not primarily HEP)", dropped, len(records), config.ARXIV_CATEGORIES)
    return kept


def main():
    nlp = spacy.load(config.SPACY_MODEL, disable=["ner"])  # keep parser+tagger+lemmatizer

    records = []
    with open(config.RAW_METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    log.info("Loaded %d records", len(records))

    records = filter_primary_category(records)
    log.info("%d records remain after primary-category filtering", len(records))
    texts = [clean_text(f"{r['title']}. {r['abstract']}") for r in records]

    n_written = 0
    with open(config.PHRASES_PATH, "w", encoding="utf-8") as out:
        for rec, doc in zip(records, nlp.pipe(texts, batch_size=64)):
            phrases = extract_phrases_from_doc(doc)
            year = int(rec["published"][:4]) if rec.get("published") else None
            out.write(json.dumps(
                {"id": rec["id"], "year": year, "phrases": phrases}, ensure_ascii=False
            ) + "\n")
            n_written += 1
    log.info("Wrote phrases for %d documents to %s", n_written, config.PHRASES_PATH)


if __name__ == "__main__":
    main()
