from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.evaluation import metrics
from ticketintel.logging import get_logger
from ticketintel.models.tfidf import TfidfClassifier
from ticketintel.reliability import calibration, conformal, novelty, selective
from ticketintel.reliability.policy import AUTO_ROUTE, decide

logger = get_logger(__name__)

# The coverage KPI the spec sets for automatic routing.
TARGET_COVERAGE = 0.60


@dataclass(frozen=True)
class FrozenPolicy:
    temperature: float
    abstention_threshold: float
    conformal_alpha: float
    conformal_qhat: float

    @classmethod
    def load(cls, settings: Settings) -> FrozenPolicy:
        raw = yaml.safe_load((settings.paths.configs / "policy.yaml").read_text(encoding="utf-8"))[
            "queue"
        ]
        return cls(
            temperature=float(raw["temperature"]),
            abstention_threshold=float(raw["abstention_threshold"]),
            conformal_alpha=float(raw["conformal_alpha"]),
            conformal_qhat=float(raw["conformal_qhat"]),
        )


def load_split(settings: Settings, split_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(settings.paths.data_processed / "tickets.parquet")
    return frame.loc[frame[schema.SPLIT] == split_name].reset_index(drop=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        revision = out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if revision == "unknown":
        return revision
    # A revision alone does not say the tree matched it. Without this suffix a
    # report produced from edited files reads as though it came from a clean
    # checkout, which is exactly the claim it cannot support.
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return revision
    return f"{revision}-dirty" if dirty.stdout.strip() else revision


def label_index(classes: np.ndarray, labels: list[str]) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[label] for label in labels])


def _risk_at_coverage(confidence: np.ndarray, correct: np.ndarray, target_coverage: float) -> float:
    # What the error rate would be if the policy were loosened until it routed
    # `target_coverage` of tickets. Reported so the trade-off against the
    # coverage KPI is a measured number rather than an assertion.
    coverage, risk = selective.risk_coverage_curve(confidence, correct)
    return float(risk[int(np.argmin(np.abs(coverage - target_coverage)))])


def evaluate_queue(settings: Settings, policy: FrozenPolicy) -> dict[str, object]:
    # The single pass over test, with the model and thresholds frozen beforehand.
    train = load_split(settings, "train")
    test = load_split(settings, "test")

    model = TfidfClassifier(c=30.0).fit(train[schema.TEXT].tolist(), train[schema.QUEUE].tolist())
    classes = model.classes_
    y_true = test[schema.QUEUE].tolist()
    y_idx = label_index(classes, y_true)

    raw_proba = model.predict_proba(test[schema.TEXT].tolist())
    proba = calibration.apply_temperature(raw_proba, policy.temperature)
    y_pred = [str(classes[i]) for i in proba.argmax(axis=1)]
    correct = np.array([int(p == t) for p, t in zip(y_pred, y_true, strict=True)])
    confidence = proba.max(axis=1)

    f1, f1_lo, f1_hi = metrics.bootstrap_ci(y_true, y_pred, metrics.macro_f1)
    worst_label, worst_recall = metrics.min_class_recall(y_true, y_pred)

    decisions = [
        decide(row, classes, policy.abstention_threshold, policy.conformal_qhat) for row in proba
    ]
    auto = np.array([d.action == AUTO_ROUTE for d in decisions])
    auto_errors = int(((~correct.astype(bool)) & auto).sum())
    auto_count = int(auto.sum())

    novel = load_split(settings, "novel_queue")
    novel_conf = calibration.apply_temperature(
        model.predict_proba(novel[schema.TEXT].tolist()), policy.temperature
    ).max(axis=1)

    return {
        "macro_f1": f1,
        "macro_f1_ci": [f1_lo, f1_hi],
        "min_class_recall": {"label": worst_label, "recall": worst_recall},
        "ece": calibration.expected_calibration_error(confidence, correct),
        "auto_coverage": auto_count / len(test),
        "auto_error_rate": (auto_errors / auto_count) if auto_count else 0.0,
        "aurc": selective.area_under_risk_coverage(confidence, correct),
        "risk_at_target_coverage": _risk_at_coverage(confidence, correct, TARGET_COVERAGE),
        "target_coverage": TARGET_COVERAGE,
        "conformal_coverage": conformal.empirical_coverage(proba, y_idx, policy.conformal_qhat),
        "conformal_mean_set_size": conformal.average_set_size(proba, policy.conformal_qhat),
        "novel_queue_auroc": novelty.novel_queue_auroc(confidence, novel_conf),
        "test_rows": len(test),
    }


def evaluate_priority(settings: Settings) -> dict[str, object]:
    train = load_split(settings, "train")
    test = load_split(settings, "test")
    model = TfidfClassifier(c=100.0).fit(
        train[schema.TEXT].tolist(), train[schema.PRIORITY].tolist()
    )
    y_true = test[schema.PRIORITY].tolist()
    proba = model.predict_proba(test[schema.TEXT].tolist())
    y_pred = [str(model.classes_[i]) for i in proba.argmax(axis=1)]

    f1, f1_lo, f1_hi = metrics.bootstrap_ci(y_true, y_pred, metrics.macro_f1)
    mae, mae_lo, mae_hi = metrics.bootstrap_ci(y_true, y_pred, metrics.ordinal_mae)
    return {
        "macro_f1": f1,
        "macro_f1_ci": [f1_lo, f1_hi],
        "ordinal_mae": mae,
        "ordinal_mae_ci": [mae_lo, mae_hi],
    }


def provenance(settings: Settings) -> dict[str, str]:
    # Everything needed to tie these numbers back to exactly what produced them.
    data_path = settings.paths.data_processed / "tickets.parquet"
    return {
        "dataset_id": settings.dataset.id,
        "dataset_revision": settings.dataset.revision,
        "processed_data_sha256": file_sha256(data_path) if data_path.exists() else "missing",
        "git_revision": git_revision(),
        "seed": str(settings.seed),
    }


def write_results(results: dict[str, object], settings: Settings) -> Path:
    settings.paths.reports.mkdir(parents=True, exist_ok=True)
    path = settings.paths.reports / "results.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return path
