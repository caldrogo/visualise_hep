"""
Shared configuration for the arXiv HEP topic-trend pipeline.
Edit these values to change scope without touching stage code.
"""

from pathlib import Path

# ---- arXiv scope -----------------------------------------------------------
# hep-ph: phenomenology, hep-th: theory, hep-ex: experiment, hep-lat: lattice
ARXIV_CATEGORIES = ["hep-ph", "hep-th", "hep-ex", "hep-lat"]

START_DATE = "2016-01-01"   # inclusive, YYYY-MM-DD
END_DATE = "2026-01-01"     # exclusive, YYYY-MM-DD

MAX_RESULTS_PER_REQUEST = 200     # arXiv API page size
REQUEST_DELAY_SECONDS = 3.0       # arXiv asks for >=3s between requests
MAX_TOTAL_RESULTS = None          # cap total records fetched, or None for no cap

# arXiv's `start` offset gets unreliable deep into a result set (large
# offsets are a known source of intermittent 500 errors) — a full year of
# hep-ph+hep-th easily exceeds this. Instead of one query per calendar year,
# fetch_arxiv.py recursively bisects the date range so no single query has
# more than this many matching results.
SAFE_MAX_RESULTS_PER_SLICE = 2000

# Retries for transient errors (500/503/connection resets), which arXiv's
# API produces occasionally even for reasonably-sized requests.
MAX_RETRIES = 6
RETRY_BACKOFF_BASE_SECONDS = 10   # wait = BASE * 2**attempt, capped below
RETRY_BACKOFF_MAX_SECONDS = 120

# ---- Phrase extraction ------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"
PHRASE_MIN_TOKENS = 2
PHRASE_MAX_TOKENS = 5
# Generic/low-signal words that survive POS filtering but carry no topical
# meaning in a physics abstract; extend as you review output.
EXTRA_STOPWORDS = {
    "result", "results", "paper", "study", "work", "approach", "method",
    "case", "term", "terms", "way", "role", "order", "context", "sample",
    "set", "value", "values", "effect", "effects", "model", "models",
    "analysis", "framework",
}

# ---- Phrase clustering (embeddings -> PCA -> HDBSCAN, scikit-learn only) ------
CLUSTER_MIN_PHRASE_COUNT = 2       # ignore phrases seen fewer times than this
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # also used by train_embedding_model.py

# PCA (sklearn.decomposition.PCA): reduces embedding dimensionality before
# HDBSCAN, since HDBSCAN's density-based clustering degrades in raw
# high-dim embedding space (distance concentration). A float n_components
# tells sklearn to keep however many components are needed to retain that
# fraction of variance, rather than a fixed dimension count. Note this is a
# linear reduction — it won't preserve non-linear local neighborhood
# structure the way UMAP does (see git history for the UMAP version of this
# file), but it's deterministic, needs no extra non-sklearn dependency, and
# was validated directly against synthetic embedding-like data before
# shipping (see the PR / conversation notes): correctly recovered known
# cluster structure through the full PCA -> HDBSCAN pipeline.
PCA_VARIANCE_RATIO = 0.95    # fraction of embedding variance to retain
PCA_RANDOM_STATE = 42        # fixes sklearn's randomized SVD solver for reproducibility

# HDBSCAN (sklearn.cluster.HDBSCAN, native since sklearn 1.3 — no separate
# `hdbscan` package needed): replaces the old kNN-graph + connected-components
# approach, which is effectively single-linkage clustering and prone to
# chaining unrelated phrases together through weak intermediate links (e.g.
# "dark matter" and "supersymmetry" ending up in one cluster). HDBSCAN's
# cluster-stability extraction avoids that, and adapts to local density
# instead of one global similarity threshold for the whole vocabulary.
HDBSCAN_MIN_CLUSTER_SIZE = 2       # a legitimate phrase cluster can be just 2 near-duplicate phrasings
HDBSCAN_MIN_SAMPLES = 1            # low = less conservative about labeling points as noise
HDBSCAN_CLUSTER_SELECTION_METHOD = "leaf"  # smaller, purer clusters; 'eom' favors fewer/larger/coarser ones
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.05    # distance floor merging leaf-splits back together; raise in
                                            # small steps (0.05, 0.1, ...) if clusters look needlessly
                                            # fragmented and re-run — sklearn's HDBSCAN doesn't expose the
                                            # condensed_tree_ the standalone hdbscan package does, so this
                                            # has to be swept empirically rather than read off the tree

# ---- Trend analysis -----------------------------------------------------------
TIME_BUCKET = "year"               # "year" or "quarter"
TREND_MIN_TOTAL_OCCURRENCES = 3    # ignore clusters with fewer total mentions
TREND_RECENT_WINDOW = 2            # number of most recent buckets = "recent"
TREND_BASELINE_MIN_BUCKETS = 2     # need at least this many earlier buckets
TOP_N_EMERGING = 25

# ---- Paths ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RAW_METADATA_PATH = DATA_DIR / "raw_metadata.jsonl"
FETCH_CHECKPOINT_PATH = DATA_DIR / "fetch_checkpoint.json"
PHRASES_PATH = DATA_DIR / "doc_phrases.jsonl"
CLUSTERS_PATH = DATA_DIR / "phrase_clusters.json"
DOC_CLUSTER_PATH = DATA_DIR / "doc_cluster_occurrences.jsonl"
EMERGING_TOPICS_CSV = RESULTS_DIR / "emerging_topics.csv"
EMERGING_TOPICS_PLOT = RESULTS_DIR / "emerging_topics.png"

# ---- Embedding fine-tuning (optional, feeds cluster_phrases.py's EMBEDDING_MODEL_NAME) ----
SYNONYM_PAIRS_PATH = DATA_DIR / "synonym_pairs.jsonl"
FINE_TUNE_BASE_MODEL = EMBEDDING_MODEL_NAME  # start from the same model cluster_phrases.py uses
FINE_TUNED_MODEL_DIR = BASE_DIR / "models" / "hep-phrase-embedder"
FINE_TUNE_EVAL_FRACTION = 0.1
FINE_TUNE_EPOCHS = 4
FINE_TUNE_BATCH_SIZE = 64

for d in (DATA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)