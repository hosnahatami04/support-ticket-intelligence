from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.evaluation import metrics
from ticketintel.evaluation.errors import top_confusions
from ticketintel.logging import get_logger
from ticketintel.models.tfidf import select_c

logger = get_logger(__name__)

TASKS = {"queue": schema.QUEUE, "priority": schema.PRIORITY}


def _load_split(settings: Settings, split_name: str) -> pd.DataFrame:
    path = settings.paths.data_processed / "tickets.parquet"
    frame = pd.read_parquet(path)
    return frame.loc[frame[schema.SPLIT] == split_name].reset_index(drop=True)


def evaluate_predictions(
    task: str, y_true: list[str], y_pred: list[str], sample_ids: list[int]
) -> dict[str, object]:
    f1_point, f1_lo, f1_hi = metrics.bootstrap_ci(y_true, y_pred, metrics.macro_f1)
    worst_label, worst_recall = metrics.min_class_recall(y_true, y_pred)
    result: dict[str, object] = {
        "task": task,
        "macro_f1": f1_point,
        "macro_f1_ci": [f1_lo, f1_hi],
        "min_class_recall": {"label": worst_label, "recall": worst_recall},
        "per_class_f1": metrics.per_class_f1(y_true, y_pred),
        "top_confusions": [vars(c) for c in top_confusions(y_true, y_pred, sample_ids)],
    }
    if task == "priority":
        mae_point, mae_lo, mae_hi = metrics.bootstrap_ci(y_true, y_pred, metrics.ordinal_mae)
        result["ordinal_mae"] = mae_point
        result["ordinal_mae_ci"] = [mae_lo, mae_hi]
    return result


def run_tfidf_baseline(settings: Settings) -> dict[str, object]:
    train = _load_split(settings, "train")
    val = _load_split(settings, "validation")
    out: dict[str, object] = {}

    settings.paths.artifacts.mkdir(parents=True, exist_ok=True)
    for task, column in TASKS.items():
        model, c_scores = select_c(
            train[schema.TEXT].tolist(),
            train[column].tolist(),
            val[schema.TEXT].tolist(),
            val[column].tolist(),
        )
        # Persist the fitted classifier so the service can load it. Without
        # this the baseline command would compute validation metrics and
        # throw the model away, leaving /triage with no artifact to open.
        model.save(settings.paths.artifacts / f"tfidf_{task}.joblib")
        proba = model.predict_proba(val[schema.TEXT].tolist())
        y_pred = model.classes_[proba.argmax(axis=1)].tolist()
        evaluation = evaluate_predictions(task, val[column].tolist(), y_pred, list(val.index))
        evaluation["c_scores"] = c_scores
        evaluation["selected_c"] = model.c
        out[task] = evaluation
    return out


def write_interim(results: dict[str, object], settings: Settings, name: str) -> Path:
    out_dir = settings.paths.reports / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _demo() -> None:  # pragma: no cover - exercised via the CLI
    from ticketintel.config import load_settings

    settings = load_settings()
    results = run_tfidf_baseline(settings)
    path = write_interim(results, settings, "baseline_tfidf_validation.json")
    logger.info("wrote interim baseline", extra={"path": str(path)})
