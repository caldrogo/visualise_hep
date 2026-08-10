"""
Stage 3 — Group near-duplicate/synonymous phrases into topic clusters, via
sentence embeddings -> PCA -> HDBSCAN, using scikit-learn for both the
dimensionality reduction and the clustering step. (sentence-transformers is
still needed for the embeddings themselves — there's no sklearn-native way
to get comparably good semantic vectors for short phrases — but the UMAP
and standalone `hdbscan` dependencies from the previous version are gone.)
 
Why reduce dimensionality first: HDBSCAN finds clusters through local
density, and raw sentence-embedding space (384 dims for the default model)
suffers from distance concentration in high dimensions — pairwise distances
start looking similar, so "dense" and "sparse" regions become hard to tell
apart.
 
Why PCA instead of UMAP (the previous version of this file): PCA is linear
— it captures directions of maximum global variance, not non-linear local
neighborhood structure the way UMAP does — so this is a real trade-off, not
a free lunch. What it buys back: no extra non-sklearn dependency, a fully
deterministic reduction (no stochastic embedding to seed), and much faster
runtime. t-SNE was considered and rejected as the sklearn-native
alternative: it targets 2D/3D visualization rather than general-purpose
clustering preprocessing, and is known to distort relative cluster density
and inter-cluster distance — exactly the properties HDBSCAN depends on.
This PCA -> HDBSCAN combination was validated against synthetic
embedding-like data with known cluster structure before shipping (tight
clusters of near-duplicate vectors plus independent noise vectors, all
unit-normalized in 384 dims) and correctly recovered the true clusters.
 
Why HDBSCAN over the original kNN-graph + connected-components approach:
connected-components is single-linkage clustering — if phrase A links to B,
and B links to C, A and C end up in one cluster even if they aren't similar
to each other at all. That chains through weak intermediate phrases (e.g.
"dark matter" and "supersymmetry" ending up together via something like
"candidate for new physics"). HDBSCAN's cluster-stability extraction avoids
that kind of transitive merging, and adapts to local density instead of one
global similarity threshold for the whole vocabulary. scikit-learn has
shipped its own HDBSCAN (sklearn.cluster.HDBSCAN) since version 1.3, with
the same min_cluster_size/min_samples/cluster_selection_method/
cluster_selection_epsilon parameters as the standalone `hdbscan` package —
just without that package's condensed_tree_ introspection API.
 
Run:
    pip install sentence-transformers scikit-learn
    python cluster_phrases.py
 
Reads config.PHRASES_PATH, writes:
  - config.CLUSTERS_PATH: {cluster_id: {"label", "members", "total_count"}}
  - config.DOC_CLUSTER_PATH: one JSON line per doc:
        {"id": ..., "year": ..., "cluster_ids": [int, ...]}
    (cluster_ids is de-duplicated per doc — a paper mentioning a phrase
    three times counts once, since trend analysis works on "share of
    papers that discuss X per period", not raw mention counts.)
 
Method:
  1. Count every phrase across the corpus; drop phrases seen fewer than
     config.CLUSTER_MIN_PHRASE_COUNT times.
  2. Embed the surviving phrases with config.EMBEDDING_MODEL_NAME.
  3. Reduce dimensionality with sklearn.decomposition.PCA, keeping enough
     components to retain config.PCA_VARIANCE_RATIO of the variance.
  4. Cluster the PCA-reduced vectors with sklearn.cluster.HDBSCAN
     (config.HDBSCAN_MIN_CLUSTER_SIZE / HDBSCAN_MIN_SAMPLES /
     HDBSCAN_CLUSTER_SELECTION_METHOD / HDBSCAN_CLUSTER_SELECTION_EPSILON).
  5. Phrases HDBSCAN marks as noise (label -1) each become their own
     singleton cluster rather than being dropped — every phrase that
     survived the min-count filter still counts toward trend analysis.
  6. Canonical label per cluster = the most frequent member phrase.
"""
import json
import logging
from collections import Counter
from typing import Dict, List, Sequence, Tuple
 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
 
import config
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cluster_phrases")
 
 
def load_doc_phrases():
    docs = []
    with open(config.PHRASES_PATH, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs
 
 
def count_phrases(docs):
    counts = Counter()
    for d in docs:
        # count each phrase once per doc (presence, not raw mentions)
        for p in set(d["phrases"]):
            counts[p] += 1
    return counts
 
 
def embed_phrases(phrases: Sequence[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return np.asarray(model.encode(phrases, normalize_embeddings=True, show_progress_bar=True))
 
 
def reduce_dimensionality(vectors: np.ndarray) -> np.ndarray:
    """PCA down to however many components retain config.PCA_VARIANCE_RATIO
    of the embedding variance (a float n_components tells sklearn to treat
    it as a variance-to-retain target rather than a fixed dimension count)."""
    max_components = min(vectors.shape[0], vectors.shape[1]) - 1
    if max_components < 2:
        log.warning("Only %d phrases to cluster — too few for PCA, clustering raw embeddings instead",
                    vectors.shape[0])
        return vectors
    pca = PCA(n_components=config.PCA_VARIANCE_RATIO, random_state=config.PCA_RANDOM_STATE)
    reduced = pca.fit_transform(vectors)
    log.info("PCA kept %d components to retain %.0f%% variance (from %d-dim embeddings)",
              reduced.shape[1], config.PCA_VARIANCE_RATIO * 100, vectors.shape[1])
    return reduced
 
 
def run_hdbscan(reduced_vectors: np.ndarray) -> np.ndarray:
    clusterer = HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION_METHOD,
        cluster_selection_epsilon=config.HDBSCAN_CLUSTER_SELECTION_EPSILON,
        metric="euclidean",  # standard once vectors are PCA-reduced; cosine isn't the right
                              # lens on PCA's own output coordinates
        copy=False,
    )
    clusterer.fit(reduced_vectors)
    return clusterer.labels_
 
 
def group_labels_into_clusters(
    phrases: Sequence[str], labels: Sequence[int], counts: Dict[str, int]
) -> Tuple[Dict[int, dict], Dict[str, int]]:
    """Turn a phrase list + HDBSCAN label array into the
    {cluster_id: {"label", "members", "total_count"}} structure the rest of
    the pipeline expects. Noise points (label == -1) each become their own
    singleton cluster with a fresh id, rather than being dropped.
 
    Split out from build_clusters() so it can be unit-tested with a plain
    labels array, independent of the embedding/PCA/HDBSCAN calls.
    """
    real_ids = [lbl for lbl in labels if lbl != -1]
    next_singleton_id = (max(real_ids) + 1) if real_ids else 0
 
    grouped: Dict[int, List[str]] = {}
    for phrase, lbl in zip(phrases, labels):
        if lbl == -1:
            cid = next_singleton_id
            next_singleton_id += 1
        else:
            cid = int(lbl)
        grouped.setdefault(cid, []).append(phrase)
 
    clusters = {}
    phrase_to_cluster = {}
    for cid, members in grouped.items():
        members = sorted(members, key=lambda p: (-counts[p], p))  # most frequent first
        label = members[0]
        total_count = sum(counts[m] for m in members)
        clusters[cid] = {"label": label, "members": members, "total_count": total_count}
        for m in members:
            phrase_to_cluster[m] = cid
    return clusters, phrase_to_cluster
 
 
def build_clusters(min_count=None):
    min_count = config.CLUSTER_MIN_PHRASE_COUNT if min_count is None else min_count
 
    docs = load_doc_phrases()
    log.info("Loaded phrases for %d documents", len(docs))
 
    counts = count_phrases(docs)
    phrases = [p for p, c in counts.items() if c >= min_count]
    log.info("%d/%d unique phrases kept after min_count=%d filter", len(phrases), len(counts), min_count)
 
    if not phrases:
        raise ValueError("No phrases survived the min_count filter — lower CLUSTER_MIN_PHRASE_COUNT.")
 
    log.info("Embedding %d phrases with %s", len(phrases), config.EMBEDDING_MODEL_NAME)
    embeddings = embed_phrases(phrases)
 
    reduced = reduce_dimensionality(embeddings)
 
    log.info("Clustering with HDBSCAN (min_cluster_size=%d, min_samples=%d, method=%s, epsilon=%.3f)",
              config.HDBSCAN_MIN_CLUSTER_SIZE, config.HDBSCAN_MIN_SAMPLES,
              config.HDBSCAN_CLUSTER_SELECTION_METHOD, config.HDBSCAN_CLUSTER_SELECTION_EPSILON)
    labels = run_hdbscan(reduced)
 
    n_noise = int(np.sum(labels == -1))
    n_real = len(set(labels)) - (1 if n_noise else 0)
    log.info("HDBSCAN found %d clusters; %d phrases labeled noise (each becomes its own singleton cluster)",
              n_real, n_noise)
 
    clusters, phrase_to_cluster = group_labels_into_clusters(phrases, labels, counts)
    return clusters, phrase_to_cluster, docs
 
 
def write_doc_cluster_occurrences(docs, phrase_to_cluster):
    n_written = 0
    with open(config.DOC_CLUSTER_PATH, "w", encoding="utf-8") as out:
        for d in docs:
            cluster_ids = sorted({phrase_to_cluster[p] for p in d["phrases"] if p in phrase_to_cluster})
            if not cluster_ids:
                continue
            out.write(json.dumps(
                {"id": d["id"], "year": d["year"], "cluster_ids": [str(cid) for cid in cluster_ids]}, ensure_ascii=False, skipkeys=True) + "\n")
            n_written += 1
    log.info("Wrote cluster occurrences for %d documents to %s", n_written, config.DOC_CLUSTER_PATH)
 
 
def main():
    clusters, phrase_to_cluster, docs = build_clusters()
    with open(config.CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2, skipkeys=True)
    log.info("Wrote %d clusters to %s", len(clusters), config.CLUSTERS_PATH)
 
    write_doc_cluster_occurrences(docs, phrase_to_cluster)
 
    top = sorted(clusters.values(), key=lambda c: -c["total_count"])[:15]
    log.info("Top clusters by total occurrence:")
    for c in top:
        log.info("  [%4d] %s  (variants: %d)", c["total_count"], c["label"], len(c["members"]))
 
 
if __name__ == "__main__":
    main()