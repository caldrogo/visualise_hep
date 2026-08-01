"""
Fine-tune a sentence-transformers embedding model on the mined synonym pairs,
so cluster_phrases.py's "embedding" backend can catch true synonyms
(e.g. "ALP" <-> "axion-like particle") that the default "tfidf" backend
structurally can't, since they share no character n-grams.

Setup (once):
    pip install sentence-transformers datasets

Run:
    python mine_synonym_pairs.py     # if you haven't already
    python train_embedding_model.py

Reads config.SYNONYM_PAIRS_PATH, fine-tunes config.FINE_TUNE_BASE_MODEL,
writes the result to config.FINE_TUNED_MODEL_DIR.

Method: MultipleNegativesRankingLoss (MNRL) — the standard choice when you
only have positive pairs and no labeled negatives. Each (anchor, positive)
pair in a batch is pulled together in embedding space, while every *other*
pair's phrases in that same batch act as free negatives (in-batch
negatives) — no negative mining required, and it's the most data-efficient
loss sentence-transformers offers for this shape of data.

To actually use the result, point cluster_phrases.py at it:
    config.CLUSTER_BACKEND = "embedding"
    config.EMBEDDING_MODEL_NAME = str(config.FINE_TUNED_MODEL_DIR)
"""
import json
import logging
import random

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_embedding_model")


def load_pairs():
    pairs = []
    with open(config.SYNONYM_PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pairs.append((d["phrase_a"], d["phrase_b"]))
    return pairs


def build_eval_set(train_pairs, eval_pairs, all_phrases, n_negatives):
    """EmbeddingSimilarityEvaluator wants (sentence1, sentence2, score) triples.
    We only mined positives, so for *evaluation only* we pad in random
    phrase-pairs as presumed negatives (score=0) — a weak assumption
    (two random phrases occasionally ARE related) but fine for tracking
    relative progress across training, which is all this is used for."""
    s1, s2, scores = [], [], []
    for a, b in eval_pairs:
        s1.append(a); s2.append(b); scores.append(1.0)
    rng = random.Random(0)
    for _ in range(n_negatives):
        a, b = rng.sample(all_phrases, 2)
        s1.append(a); s2.append(b); scores.append(0.0)
    return s1, s2, scores


def main():
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

    pairs = load_pairs()
    log.info("Loaded %d synonym pairs", len(pairs))
    if len(pairs) < 50:
        log.warning("Very few pairs (%d) — fine-tuning quality will be limited. "
                    "Consider widening the mining net (e.g. lower CLUSTER_MIN_PHRASE_COUNT "
                    "before running cluster_phrases.py, so more same-cluster pairs are "
                    "available) before investing in a long training run.", len(pairs))

    random.seed(42)
    random.shuffle(pairs)
    n_eval = max(1, int(len(pairs) * config.FINE_TUNE_EVAL_FRACTION))
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]

    train_dataset = Dataset.from_dict({
        "anchor": [a for a, _ in train_pairs],
        "positive": [b for _, b in train_pairs],
    })

    all_phrases = list({p for pair in pairs for p in pair})
    s1, s2, scores = build_eval_set(train_pairs, eval_pairs, all_phrases, n_negatives=len(eval_pairs))
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=s1, sentences2=s2, scores=scores, name="hep-synonym-pairs-eval"
    )

    model = SentenceTransformer(config.FINE_TUNE_BASE_MODEL)
    loss = losses.MultipleNegativesRankingLoss(model)

    log.info("Baseline (pre-fine-tuning) eval score: %.4f", evaluator(model))

    args = SentenceTransformerTrainingArguments(
        output_dir=str(config.FINE_TUNED_MODEL_DIR),
        num_train_epochs=config.FINE_TUNE_EPOCHS,
        per_device_train_batch_size=config.FINE_TUNE_BATCH_SIZE,
        # MNRL treats every other in-batch example as a negative, so a
        # duplicate positive in the same batch would falsely look like a
        # negative for another pair — NO_DUPLICATES avoids that.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,
        run_name="hep-phrase-embedder",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    log.info("Final eval score: %.4f", evaluator(model))
    model.save_pretrained(str(config.FINE_TUNED_MODEL_DIR))
    log.info("Saved fine-tuned model to %s", config.FINE_TUNED_MODEL_DIR)
    log.info("To use it: set config.CLUSTER_BACKEND = \"embedding\" and "
             "config.EMBEDDING_MODEL_NAME = \"%s\"", config.FINE_TUNED_MODEL_DIR)


if __name__ == "__main__":
    main()
