from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.logging import get_logger
from ticketintel.models.tfidf import TfidfClassifier
from ticketintel.reliability import calibration, conformal, novelty, selective

logger = get_logger(__name__)

TARGET_AUTO_RISK = 0.05
# Reported alongside the operating point so the coverage/error trade-off is
# visible when both KPI targets cannot hold at once (section 4.1).
TRADEOFF_RISKS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
CONFORMAL_ALPHA = 0.10


def _load_split(settings: Settings, split_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(settings.paths.data_processed / "tickets.parquet")
    return frame.loc[frame[schema.SPLIT] == split_name].reset_index(drop=True)


def _label_index(classes: np.ndarray, labels: list[str]) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[label] for label in labels])


def calibrate_queue(settings: Settings) -> dict[str, float]:
    # All thresholds are fit on the calibration split only; test is never read.
    train = _load_split(settings, "train")
    cal = _load_split(settings, "calibration")

    model = TfidfClassifier(c=30.0).fit(train[schema.TEXT].tolist(), train[schema.QUEUE].tolist())
    classes = model.classes_
    cal_proba = model.predict_proba(cal[schema.TEXT].tolist())
    y_idx = _label_index(classes, cal[schema.QUEUE].tolist())

    correct = (cal_proba.argmax(axis=1) == y_idx).astype(int)
    conf_before = cal_proba.max(axis=1)
    ece_before = calibration.expected_calibration_error(conf_before, correct)

    temperature = calibration.fit_temperature(cal_proba, y_idx)
    cal_proba_t = calibration.apply_temperature(cal_proba, temperature)
    conf_after = cal_proba_t.max(axis=1)
    ece_after = calibration.expected_calibration_error(conf_after, correct)

    tau, coverage = selective.select_threshold_for_risk(conf_after, correct, TARGET_AUTO_RISK)
    aurc = selective.area_under_risk_coverage(conf_after, correct)
    tradeoff = {
        f"coverage_at_risk_{int(r * 100):02d}": selective.select_threshold_for_risk(
            conf_after, correct, r
        )[1]
        for r in TRADEOFF_RISKS
    }

    cal_scores = conformal.lac_scores(cal_proba_t, y_idx)
    qhat = conformal.conformal_quantile(cal_scores, CONFORMAL_ALPHA)

    # Novel-queue probe: confidence should separate familiar from novel tickets.
    novel = _load_split(settings, "novel_queue")
    novel_proba_t = calibration.apply_temperature(
        model.predict_proba(novel[schema.TEXT].tolist()), temperature
    )
    auroc = novelty.novel_queue_auroc(conf_after, novel_proba_t.max(axis=1))

    results: dict[str, float] = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "abstention_threshold": tau,
        "coverage_at_threshold": coverage,
        "aurc": aurc,
        "conformal_alpha": CONFORMAL_ALPHA,
        "conformal_qhat": qhat,
        "novel_queue_auroc": auroc,
        **tradeoff,
    }
    logger.info("calibrated queue model", extra=results)
    return results


def freeze_policy(results: dict[str, float], settings: Settings) -> Path:
    path = settings.paths.configs / "policy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = {
        "queue": {
            "temperature": results["temperature"],
            "abstention_threshold": results["abstention_threshold"],
            "conformal_alpha": results["conformal_alpha"],
            "conformal_qhat": results["conformal_qhat"],
        }
    }
    path.write_text(yaml.safe_dump(policy, sort_keys=True), encoding="utf-8")
    return path


def write_report(results: dict[str, float], settings: Settings) -> Path:
    out_dir = settings.paths.reports / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration_queue.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return path
