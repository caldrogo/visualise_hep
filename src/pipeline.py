"""
Orchestrator — run all four stages in order, or a chosen subset.

Usage:
    python pipeline.py                 # run everything
    python pipeline.py --from cluster  # skip fetch+extract, reuse existing data/
    python pipeline.py --only fetch    # run just one stage
"""
import argparse
import importlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")

# Each stage's module is imported lazily, only when that stage actually runs,
# so `--only cluster` doesn't require spacy/sentence-transformers to be
# installed just because extract_phrases.py happens to import them.
STAGE_MODULES = {
    "fetch": "fetch_arxiv",
    "extract": "extract_phrases",
    "cluster": "cluster_phrases",
    "trend": "trend_analysis",
}
STAGE_NAMES = list(STAGE_MODULES.keys())


def _run_stage(name):
    module = importlib.import_module(STAGE_MODULES[name])
    module.main()


def run(stages_to_run):
    for name in STAGE_NAMES:
        if name not in stages_to_run:
            continue
        log.info("=" * 60)
        log.info("STAGE: %s", name)
        log.info("=" * 60)
        _run_stage(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_stage", choices=STAGE_NAMES,
                         help="Run this stage and every stage after it.")
    parser.add_argument("--only", dest="only_stage", choices=STAGE_NAMES,
                         help="Run only this single stage.")
    args = parser.parse_args()

    if args.only_stage:
        stages_to_run = {args.only_stage}
    elif args.from_stage:
        idx = STAGE_NAMES.index(args.from_stage)
        stages_to_run = set(STAGE_NAMES[idx:])
    else:
        stages_to_run = set(STAGE_NAMES)

    run(stages_to_run)


if __name__ == "__main__":
    main()
