# Visualise HEP

[!Screenshot](visualise_hep.png)

Four-stage pipeline: download HEP (high-energy physics) paper metadata from
arXiv → extract candidate topic phrases from titles/abstracts → group
near-duplicate phrases into topic clusters → rank clusters by growth in
relative frequency to surface emerging research areas.

The Plotly Dash app is deployed here https://3e38fea6-fe42-4acc-9db1-2496573e0b93.plotly.app/

```
fetch_arxiv.py  →  extract_phrases.py  →  cluster_phrases.py  →  trend_analysis.py
  (arXiv API)       (spaCy noun chunks)     (similarity graph)     (trend)
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
python src/pipeline.py                 # run all four stages end to end
python src/pipeline.py --from cluster  # re-run clustering + trend analysis only
python src/pipeline.py --only trend    # just re-score, e.g. after tuning config.py
```

Or run stages individually (`python fetch_arxiv.py`, etc.) — each reads/writes
plain JSON files in `data/` and `results/`, so you can inspect intermediate
output at every step.

All scope/behavior knobs (date range, categories, thresholds) live in
**`config.py`** — nothing else needs editing for normal use.

## What each stage does

**1. `fetch_arxiv.py`** — Calls the public arXiv API
(`http://export.arxiv.org/api/query`), restricted to categories (edit `config.ARXIV_CATEGORIES`) from `config.START_DATE`
to `config.END_DATE`. Queries one year at a time and paginates within each
year. Requests are spaced over 3 seconds apart. Writes `data/raw_metadata.jsonl`.

**2. `extract_phrases.py`** — For each paper, runs spaCy over `title. abstract`
(after stripping inline LaTeX like `$m_\chi$`), pulls noun chunks of 2–5
tokens, trims leading determiners/prepositions, lemmatizes (while preserving
short ALL-CAPS acronyms like `QCD`/`LHC`/`CKM`), and drops chunks headed by a
generic word (`config.EXTRA_STOPWORDS`). Writes `data/doc_phrases.jsonl`.
This is intentionally a lightweight extractor, not a keyphrase model — it
overgenerates phrases on purpose, and stage 3 is what cleans that up. If you
have `scispacy` available, swap `config.SPACY_MODEL` to one of its models
(e.g. `en_core_sci_sm`) for much better handling of physics-specific jargon.

**3. `cluster_phrases.py`** — Real research phrasing is inconsistent
("dark matter direct detection" vs. "direct detection of dark matter" vs.
"direct detection experiments for dark matter"). This stage:
  - drops phrases seen fewer than `CLUSTER_MIN_PHRASE_COUNT` times (long tail
    of one-off phrasings — noise for trend purposes anyway),
  - vectorizes the rest and links each phrase to its nearest neighbors above
    a cosine-similarity threshold, then takes connected components as
    clusters (scales much better than full hierarchical clustering),
  - picks the most frequent member as each cluster's canonical label.

  Two backends (`config.CLUSTER_BACKEND`):
  - `"tfidf"` (default) — character n-gram TF-IDF. Fully offline, no
    downloads, and merges word-order/morphology variants well, but won't
    connect true synonyms that don't share spelling (e.g. "collider" vs.
    "accelerator", or "ALP" vs. "axion-like particle").
  - `"embedding"` — sentence-transformers (`all-MiniLM-L6-v2` by default).
    Needs `pip install sentence-transformers` and a one-time model download,
    but captures real semantic synonymy the TF-IDF backend misses. Worth
    switching to once you've confirmed the pipeline runs end to end.

  Writes `data/phrase_clusters.json` (cluster → label + member phrases) and
  `data/doc_cluster_occurrences.jsonl` (per paper, which clusters it touches —
  de-duplicated per paper, since trend analysis cares about *how many papers*
  discuss a topic, not how many times a phrase is repeated within one paper).

**4. `trend_analysis.py`** — The key methodological point: arXiv submission
volume grows every year, so raw mention counts go up for almost every topic
regardless of whether it's actually "emerging." This stage instead tracks
each cluster's **share of that year's papers** (mentions / total papers that
year), then compares a recent window (`TREND_RECENT_WINDOW` years, default 2)
against an earlier baseline window — a growth ratio of `3.0x` means the topic
is three times more prevalent in recent papers than in the baseline period.
A linear trend slope is also reported as a smoother secondary signal.
Clusters below `TREND_MIN_TOTAL_OCCURRENCES` are dropped (a phrase going from
1 to 3 mentions isn't a trend). Clusters with zero baseline occurrences are
flagged `is_new` rather than given a misleading infinite ratio.

  Writes `results/emerging_topics.csv` (full ranked table, one row per
  cluster, with per-year counts) and `results/emerging_topics.png` (trend
  lines for the top `TOP_N_EMERGING` clusters).

## Fine-tuning your own embedding model (optional)

The `embedding` cluster backend defaults to a general-purpose model
(`all-MiniLM-L6-v2`), which misses HEP-specific synonymy that shares no
spelling with itself — e.g. "ALP" and "axion-like particle". Two extra
scripts fine-tune a model on synonym pairs mined straight from your own
downloaded abstracts, no manual labeling required:

```bash
pip install sentence-transformers datasets
python mine_synonym_pairs.py      # mines acronym-definition pairs, e.g.
                                   #   "weakly interacting massive particles" <-> "WIMPs"
python train_embedding_model.py   # fine-tunes config.FINE_TUNE_BASE_MODEL on them
```

This works because HEP abstracts constantly spell out acronyms on first
use ("...effective field theory (EFT) methods..."), and every one of those
is a free, genuinely-semantic positive training pair — mined with a
Schwartz-Hearst-style heuristic in `mine_synonym_pairs.py` (skim the
mined `data/synonym_pairs.jsonl` before training; it's a heuristic, not a
certainty, and a wrong pair teaches the model false synonymy). If you've
already run `cluster_phrases.py`, it also pulls in same-cluster phrases as
supplementary positives.

Training uses `MultipleNegativesRankingLoss` — the standard choice when you
only have positive pairs and no labeled negatives; other pairs in the same
batch serve as free negatives. Once trained, point the pipeline at it:

```python
# config.py
CLUSTER_BACKEND = "embedding"
EMBEDDING_MODEL_NAME = str(FINE_TUNED_MODEL_DIR)  # or the literal path printed at the end of training
```

then re-run `python pipeline.py --from cluster` to re-cluster and re-score
with the fine-tuned model.

## Tuning

| Symptom | Fix |
|---|---|
| Too many near-duplicate topics in results | Lower `CLUSTER_SIMILARITY_THRESHOLD` (more merging), or switch to the `embedding` backend |
| Clusters merging unrelated phrases | Raise `CLUSTER_SIMILARITY_THRESHOLD` |
| Results dominated by noise / one-off phrases | Raise `CLUSTER_MIN_PHRASE_COUNT` and/or `TREND_MIN_TOTAL_OCCURRENCES` |
| Missing obviously-relevant phrases | Extend or trim `EXTRA_STOPWORDS`; widen `PHRASE_MIN_TOKENS`/`PHRASE_MAX_TOKENS` |
| Want quarterly instead of yearly resolution | Set `TIME_BUCKET = "quarter"` and adjust `_year_ranges`/date bucketing in `trend_analysis.py` accordingly (currently wired for yearly buckets) |

## Validation note

This environment doesn't have outbound network access, so `fetch_arxiv.py`
and the spaCy-dependent `extract_phrases.py` couldn't be executed live here —
but the arXiv API query format was verified against arXiv's own API manual,
and stages 3–4 (the core clustering + trend-scoring logic) were tested
end-to-end against synthetic phrase data with two deliberately-engineered
"emerging" topics, both of which the pipeline correctly surfaced at the top
of the ranking. Run `python pipeline.py` in an environment with internet
access to go end to end on real data.
