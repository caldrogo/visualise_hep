"""
Stage 3 — Group near-duplicate/synonymous phrases into topic clusters.

Why: phrase extraction produces many surface variants of the same idea
("dark matter direct detection", "direct detection of dark matter",
"direct detection experiments for dark matter"). Counting these separately
would understate real topic frequency and hide trends. This stage merges
them into clusters and picks one canonical label per cluster.

Run:
    python cluster_phrases.py

Reads config.PHRASES_PATH, writes:
  - config.CLUSTERS_PATH: {cluster_id: {"label": str, "members": [phrase, ...],
                                          "total_count": int}}
  - config.DOC_CLUSTER_PATH: one JSON line per doc:
        {"id": ..., "year": ..., "cluster_ids": [int, ...]}
    (cluster_ids is de-duplicated per doc — a paper mentioning a phrase
    three times counts once, since trend analysis works on "share of
    papers that discuss X per period", not raw mention counts.)

Method:
  1. Count every phrase across the corpus; drop phrases seen fewer than
     config.CLUSTER_MIN_PHRASE_COUNT times (long tail of near-unique
     phrasings — clustering noise, and irrelevant to trend detection anyway).
  2. Vectorize the surviving phrases (config.CLUSTER_BACKEND):
       - "tfidf": character n-gram TF-IDF + cosine similarity. Fully offline,
         robust to word-order/morphology variants, no model download.
       - "embedding": sentence-transformers embeddings + cosine similarity.
         Captures true synonyms (e.g. "collider" vs "accelerator") that
         TF-IDF can't, but needs `pip install sentence-transformers` and a
         one-time model download.
  3. Build a similarity graph: link phrase i and j if j is among i's top-k
     nearest neighbors AND similarity >= config.CLUSTER_SIMILARITY_THRESHOLD.
     Take connected components as clusters. This scales far better than
     full hierarchical clustering on large vocabularies.
  4. Canonical label = most frequent member phrase in the cluster.
"""
import json
import logging
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import networkx as nx

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


def vectorize_tfidf(phrases):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1)
    return vec.fit_transform(phrases)  # sparse matrix, rows = phrases


def vectorize_embedding(phrases):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return model.encode(phrases, normalize_embeddings=True, show_progress_bar=True)


def build_similarity_graph(phrases, vectors, k, threshold):
    n = len(phrases)
    k = min(k + 1, n)  # +1 because a point is its own nearest neighbor
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            similarity = 1.0 - dist
            if similarity >= threshold:
                g.add_edge(i, j, weight=similarity)
    return g


def build_clusters(min_count=None, backend=None, threshold=None, knn=None):
    min_count = config.CLUSTER_MIN_PHRASE_COUNT if min_count is None else min_count
    backend = config.CLUSTER_BACKEND if backend is None else backend
    threshold = config.CLUSTER_SIMILARITY_THRESHOLD if threshold is None else threshold
    knn = config.CLUSTER_KNN if knn is None else knn

    docs = load_doc_phrases()
    log.info("Loaded phrases for %d documents", len(docs))

    counts = count_phrases(docs)
    phrases = [p for p, c in counts.items() if c >= min_count]
    log.info("%d/%d unique phrases kept after min_count=%d filter", len(phrases), len(counts), min_count)

    if not phrases:
        raise ValueError("No phrases survived the min_count filter — lower CLUSTER_MIN_PHRASE_COUNT.")

    if backend == "tfidf":
        vectors = vectorize_tfidf(phrases)
    elif backend == "embedding":
        vectors = vectorize_embedding(phrases)
    else:
        raise ValueError(f"Unknown CLUSTER_BACKEND: {backend}")

    graph = build_similarity_graph(phrases, vectors, knn, threshold)
    components = list(nx.connected_components(graph))
    log.info("Formed %d clusters from %d phrases", len(components), len(phrases))

    phrase_to_cluster = {}
    clusters = {}
    for cid, comp in enumerate(components):
        members = [phrases[i] for i in comp]
        members.sort(key=lambda p: (-counts[p], p))  # most frequent first
        label = members[0]
        total_count = sum(counts[m] for m in members)
        clusters[cid] = {"label": label, "members": members, "total_count": total_count}
        for m in members:
            phrase_to_cluster[m] = cid

    return clusters, phrase_to_cluster, docs


def write_doc_cluster_occurrences(docs, phrase_to_cluster):
    n_written = 0
    with open(config.DOC_CLUSTER_PATH, "w", encoding="utf-8") as out:
        for d in docs:
            cluster_ids = sorted({phrase_to_cluster[p] for p in d["phrases"] if p in phrase_to_cluster})
            if not cluster_ids:
                continue
            out.write(json.dumps(
                {"id": d["id"], "year": d["year"], "cluster_ids": cluster_ids}
            ) + "\n")
            n_written += 1
    log.info("Wrote cluster occurrences for %d documents to %s", n_written, config.DOC_CLUSTER_PATH)


def main():
    clusters, phrase_to_cluster, docs = build_clusters()
    with open(config.CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    log.info("Wrote %d clusters to %s", len(clusters), config.CLUSTERS_PATH)

    write_doc_cluster_occurrences(docs, phrase_to_cluster)

    top = sorted(clusters.values(), key=lambda c: -c["total_count"])[:15]
    log.info("Top clusters by total occurrence:")
    for c in top:
        log.info("  [%4d] %s  (variants: %d)", c["total_count"], c["label"], len(c["members"]))


if __name__ == "__main__":
    main()
