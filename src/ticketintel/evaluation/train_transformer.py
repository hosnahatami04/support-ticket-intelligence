from __future__ import annotations

import time

import pandas as pd

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.evaluation import compare
from ticketintel.logging import get_logger
from ticketintel.models.embeddings import EmbeddingClassifier
from ticketintel.models.tfidf import TfidfClassifier, select_c
from ticketintel.models.transformer import TrainConfig, TransformerClassifier

logger = get_logger(__name__)


def _fit_tfidf(train: pd.DataFrame, val: pd.DataFrame, column: str) -> TfidfClassifier:
    model, _ = select_c(
        train[schema.TEXT].tolist(),
        train[column].tolist(),
        val[schema.TEXT].tolist(),
        val[column].tolist(),
    )
    return model


def run(
    settings: Settings,
    time_budget_s: float = 3 * 60 * 60,
    max_length: int = 256,
    train_subsample: int | None = None,
) -> list[dict[str, object]]:
    train = compare._load_split(settings, "train")
    val = compare._load_split(settings, "validation")
    rows: list[dict[str, object]] = []

    # On a CPU budget a full DistilBERT epoch over the whole train split costs
    # ~90 min, which leaves room for only one epoch. Subsampling and a shorter
    # max_length let the transformer reach several epochs for a fair comparison
    # (section 4.2 / ADR 0005).
    bert_train = train
    if train_subsample is not None and train_subsample < len(train):
        bert_train = train.sample(n=train_subsample, random_state=settings.seed)

    for task, column in compare.TASKS.items():
        train_texts, train_labels = train[schema.TEXT].tolist(), train[column].tolist()
        val_texts, val_labels = val[schema.TEXT].tolist(), val[column].tolist()
        bert_texts = bert_train[schema.TEXT].tolist()
        bert_task_labels = bert_train[column].tolist()

        # Tier 1: TF-IDF.
        t0 = time.monotonic()
        tfidf = _fit_tfidf(train, val, column)
        tfidf_train_s = time.monotonic() - t0
        tfidf_path = settings.paths.artifacts / f"tfidf_{task}.joblib"
        tfidf.save(tfidf_path)
        m = compare.evaluate_model_on_validation(tfidf, val_texts, val_labels)
        rows.append(
            {
                "task": task,
                "model": "tfidf",
                "train_seconds": tfidf_train_s,
                "size_mb": compare.dir_size_mb(tfidf_path),
                **m,
            }
        )

        # Tier 2: sentence embeddings.
        t0 = time.monotonic()
        emb = EmbeddingClassifier(settings.paths.artifacts / "embedding_cache")
        emb.fit(train_texts, train_labels)
        emb_train_s = time.monotonic() - t0
        emb_path = settings.paths.artifacts / f"embeddings_{task}.joblib"
        emb.save(emb_path)
        m = compare.evaluate_model_on_validation(emb, val_texts, val_labels)
        rows.append(
            {
                "task": task,
                "model": "embeddings",
                "train_seconds": emb_train_s,
                "size_mb": compare.dir_size_mb(emb_path),
                **m,
            }
        )

        # Tier 3: DistilBERT. Split the remaining budget across the two tasks.
        t0 = time.monotonic()
        config = TrainConfig(
            time_budget_s=time_budget_s / len(compare.TASKS), max_length=max_length
        )
        bert = TransformerClassifier(config)
        bert.fit(bert_texts, bert_task_labels, val_texts, val_labels)
        bert_train_s = time.monotonic() - t0
        bert_path = settings.paths.artifacts / f"transformer_{task}"
        bert.save(bert_path)
        m = compare.evaluate_model_on_validation(bert, val_texts, val_labels)
        rows.append(
            {
                "task": task,
                "model": "transformer",
                "train_seconds": bert_train_s,
                "size_mb": compare.dir_size_mb(bert_path),
                **m,
            }
        )

        logger.info("tier comparison done", extra={"task": task})

    return rows
