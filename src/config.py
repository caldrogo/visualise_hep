"""
Shared configuration for the arXiv HEP topic-trend pipeline.
Edit these values to change scope without touching stage code.
"""

from pathlib import Path

# ---- arXiv scope -----------------------------------------------------------
# hep-ph: phenomenology, hep-th: theory, hep-ex: experiment, hep-lat: lattice
ARXIV_CATEGORIES = ["hep-ph", "hep-th"]

START_DATE = "2021-07-01"   # inclusive, YYYY-MM-DD
END_DATE = "2026-07-01"     # exclusive, YYYY-MM-DD

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

# ---- Phrase clustering -------------------------------------------------------
CLUSTER_MIN_PHRASE_COUNT = 3       # ignore phrases seen fewer times than this
CLUSTER_BACKEND = "embedding"          # "tfidf" (offline) or "embedding" (needs internet)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CLUSTER_SIMILARITY_THRESHOLD = 0.7  # cosine similarity to link two phrases
CLUSTER_KNN = 10                     # neighbors considered per phrase when linking

# ---- Trend analysis -----------------------------------------------------------
TIME_BUCKET = "year"               # "year" or "quarter"
TREND_MIN_TOTAL_OCCURRENCES = 8    # ignore clusters with fewer total mentions
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

# ---- Embedding fine-tuning (optional, feeds CLUSTER_BACKEND = "embedding") ----
SYNONYM_PAIRS_PATH = DATA_DIR / "synonym_pairs.jsonl"
FINE_TUNE_BASE_MODEL = EMBEDDING_MODEL_NAME  # start from the same model cluster_phrases.py uses
FINE_TUNED_MODEL_DIR = BASE_DIR / "models" / "hep-phrase-embedder"
FINE_TUNE_EVAL_FRACTION = 0.1
FINE_TUNE_EPOCHS = 4
FINE_TUNE_BATCH_SIZE = 64

for d in (DATA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
