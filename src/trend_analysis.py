"""
Stage 4 — Find emerging research areas from clustered phrase occurrences.

Run:
    python trend_analysis.py

Reads config.CLUSTERS_PATH + config.DOC_CLUSTER_PATH + config.PHRASES_PATH,
writes config.EMERGING_TOPICS_CSV (ranked table) and config.EMERGING_TOPICS_PLOT
(trend lines for the top emerging clusters).

Method — why not just "count went up":
  Raw mention counts go up for almost everything simply because arXiv
  submission volume grows every year. "Emerging" should mean a topic's
  SHARE of the literature is growing, not just its absolute count. So:

  1. For each cluster, compute per-year share = (# papers mentioning the
     cluster that year) / (# papers in the corpus that year).
  2. Emergence score = recent-period average share vs. earlier-period
     average share (config.TREND_RECENT_WINDOW / TREND_BASELINE_MIN_BUCKETS),
     i.e. a growth ratio. A ratio of 3.0 means "3x more prevalent recently
     than in the earlier baseline window."
  3. We also fit a linear trend (share vs. year) as a second, smoother
     signal, and report both.
  4. Clusters with too few total occurrences are dropped
     (config.TREND_MIN_TOTAL_OCCURRENCES) — a phrase that jumped from 1
     mention to 3 mentions is not a signal, it's noise.
  5. Clusters with zero occurrences in the baseline window are flagged
     "new" rather than given an infinite ratio.
"""
import json
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trend_analysis")

EPS = 1e-6


def load_data():
    with open(config.CLUSTERS_PATH, encoding="utf-8") as f:
        clusters = json.load(f)  # {str(cluster_id): {"label", "members", "total_count"}}

    doc_years = Counter()
    with open(config.PHRASES_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("year"):
                doc_years[d["year"]] += 1

    cluster_year_counts = defaultdict(Counter)  # cluster_id (int) -> year -> doc count
    with open(config.DOC_CLUSTER_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            year = d.get("year")
            if not year:
                continue
            for cid in d["cluster_ids"]:
                cluster_year_counts[cid][year] += 1

    return clusters, doc_years, cluster_year_counts


def _year_range(doc_years):
    return list(range(min(doc_years), max(doc_years) + 1))


def compute_emergence(clusters, doc_years, cluster_year_counts):
    years = _year_range(doc_years)
    n_recent = config.TREND_RECENT_WINDOW
    n_baseline_min = config.TREND_BASELINE_MIN_BUCKETS

    rows = []
    for cid_str, meta in clusters.items():
        cid = int(cid_str)
        year_counts = cluster_year_counts.get(cid, Counter())
        total = sum(year_counts.get(y, 0) for y in years)
        if total < config.TREND_MIN_TOTAL_OCCURRENCES:
            continue

        shares = np.array([year_counts.get(y, 0) / doc_years[y] for y in years])

        if len(years) < n_recent + n_baseline_min:
            # not enough history to split into baseline/recent meaningfully
            recent_avg = shares[-n_recent:].mean()
            baseline_avg = shares[:-n_recent].mean() if len(years) > n_recent else 0.0
        else:
            recent_avg = shares[-n_recent:].mean()
            baseline_avg = shares[:-n_recent].mean()

        is_new = baseline_avg == 0 and recent_avg > 0
        growth_ratio = (recent_avg + EPS) / (baseline_avg + EPS)

        # linear trend of share over time (smoother, less sensitive to a single spike)
        slope = float(np.polyfit(years, shares, 1)[0]) if len(years) >= 2 else 0.0

        row = {
            "cluster_id": cid,
            "label": meta["label"],
            "n_variants": len(meta["members"]),
            "total_occurrences": total,
            "recent_avg_share_pct": round(recent_avg * 100, 4),
            "baseline_avg_share_pct": round(baseline_avg * 100, 4),
            "growth_ratio": round(growth_ratio, 3),
            "is_new": is_new,
            "trend_slope_pct_per_year": round(slope * 100, 5),
        }
        for y in years:
            row[f"y{y}"] = year_counts.get(y, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["is_new", "growth_ratio"], ascending=[False, False]).reset_index(drop=True)


def plot_top_emerging(df, doc_years, top_n, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = _year_range(doc_years)
    year_cols = [f"y{y}" for y in years]
    top = df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in top.iterrows():
        shares = [row[c] / doc_years[y] * 100 for c, y in zip(year_cols, years)]
        ax.plot(years, shares, marker="o", label=row["label"])

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of HEP papers mentioning topic (%)")
    ax.set_title(f"Top {len(top)} emerging HEP topics (arXiv {config.ARXIV_CATEGORIES})")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info("Saved trend plot to %s", out_path)


def main():
    clusters, doc_years, cluster_year_counts = load_data()
    log.info("Loaded %d clusters across years %s", len(clusters), sorted(doc_years))

    df = compute_emergence(clusters, doc_years, cluster_year_counts)
    if df.empty:
        log.warning("No clusters passed TREND_MIN_TOTAL_OCCURRENCES=%d — lower the threshold.",
                    config.TREND_MIN_TOTAL_OCCURRENCES)
        return

    df.to_csv(config.EMERGING_TOPICS_CSV, index=False)
    log.info("Wrote %d ranked clusters to %s", len(df), config.EMERGING_TOPICS_CSV)

    plot_top_emerging(df, doc_years, config.TOP_N_EMERGING, config.EMERGING_TOPICS_PLOT)

    log.info("Top %d emerging HEP topics:", config.TOP_N_EMERGING)
    cols = ["label", "total_occurrences", "baseline_avg_share_pct", "recent_avg_share_pct", "growth_ratio", "is_new"]
    for _, row in df.head(config.TOP_N_EMERGING)[cols].iterrows():
        tag = "NEW" if row["is_new"] else f'{row["growth_ratio"]}x'
        log.info("  %-45s total=%-4d baseline=%.3f%% -> recent=%.3f%%  (%s)",
                  row["label"], row["total_occurrences"], row["baseline_avg_share_pct"],
                  row["recent_avg_share_pct"], tag)


if __name__ == "__main__":
    main()
