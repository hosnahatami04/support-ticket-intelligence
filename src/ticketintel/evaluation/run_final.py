from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.evaluation import ablations, final, report
from ticketintel.logging import get_logger
from ticketintel.models.tfidf import TfidfClassifier
from ticketintel.reliability import calibration, novelty

logger = get_logger(__name__)

# Findings that the measurements themselves established. Kept here so the
# report states them in the same run that produces the numbers.
WHAT_DIDNT_WORK = [
    "- A fine-tuned DistilBERT did not beat the TF-IDF baseline within the CPU",
    "  budget. Given several epochs on a shortened, subsampled corpus it reached",
    "  macro-F1 0.245 on queue against 0.578 for TF-IDF, at 268 MB and roughly",
    "  fifteen times the latency. The lighter model was selected",
    "  (docs/decisions/0005).",
    "- Sentence embeddings also underperformed the baseline on both tasks",
    "  (queue 0.292, priority 0.437), so the middle tier earned nothing either.",
    "- Character n-grams did not help. On the final comparison word-only scored",
    "  marginally higher than word plus character n-grams, so the extra feature",
    "  space bought nothing measurable on this data.",
    "- The conformal set barely changes the routing decision. Threshold-only and",
    "  threshold-plus-conformal give nearly identical coverage and error, because",
    "  the abstention threshold is already strict enough that the singleton",
    "  condition rarely binds. The guarantee is still worth reporting, but it is",
    "  not what is holding the error rate down.",
    "- The 5% auto-route error target and the 60% coverage target cannot hold at",
    "  once on this dataset. The error target is met comfortably, but only a",
    "  small share of tickets clear the bar; reaching 60% coverage would cost",
    "  roughly 27% error. The trade-off is reported rather than tuned away.",
    "- Max calibrated probability barely separates the held-out queue from known",
    "  ones, so novel-queue AUROC falls well short of 0.75. Confidence alone is a",
    "  weak novelty signal here.",
    "- Temperature scaling improved calibration but not enough to clear the 0.03",
    "  ECE target on test.",
    "- The first faithfulness measurement scored 0.22 because the summary fields",
    "  are noun phrases and an entailment model scores propositions. Restating",
    "  each field as a declarative claim raised it to 0.42. That was a fix to the",
    "  measurement, not to the summaries.",
]


def run(settings: Settings) -> dict[str, Any]:
    policy = final.FrozenPolicy.load(settings)
    logger.info("loaded frozen policy", extra={"threshold": policy.abstention_threshold})

    queue = final.evaluate_queue(settings, policy)
    priority = final.evaluate_priority(settings)
    abl = ablations.run_all(settings, policy)

    results: dict[str, Any] = {
        "queue": queue,
        "priority": priority,
        "ablations": abl,
        "provenance": final.provenance(settings),
        "what_didnt_work": WHAT_DIDNT_WORK,
    }

    summarization = settings.paths.reports / "interim" / "summarization_eval.json"
    if summarization.exists():
        results["summarization"] = report.load_json(summarization)

    return results


def make_figures(settings: Settings) -> list[Path]:
    # Refits the frozen-configuration model once more to get the arrays the
    # plots need; the metrics themselves come from the evaluation above.
    policy = final.FrozenPolicy.load(settings)
    train = final.load_split(settings, "train")
    test = final.load_split(settings, "test")

    model = TfidfClassifier(c=30.0).fit(train[schema.TEXT].tolist(), train[schema.QUEUE].tolist())
    classes = model.classes_
    y_true = test[schema.QUEUE].tolist()
    raw = model.predict_proba(test[schema.TEXT].tolist())
    scaled = calibration.apply_temperature(raw, policy.temperature)
    y_pred = [str(classes[i]) for i in scaled.argmax(axis=1)]
    correct = np.array([int(p == t) for p, t in zip(y_pred, y_true, strict=True)])

    novel = final.load_split(settings, "novel_queue")
    novel_conf = calibration.apply_temperature(
        model.predict_proba(novel[schema.TEXT].tolist()), policy.temperature
    ).max(axis=1)
    auroc_value = novelty.novel_queue_auroc(scaled.max(axis=1), novel_conf)

    return [
        report.plot_confusion_matrix(y_true, y_pred, settings),
        report.plot_reliability(raw.max(axis=1), scaled.max(axis=1), correct, settings),
        report.plot_risk_coverage(
            {
                "tfidf (uncalibrated)": (raw.max(axis=1), correct),
                "tfidf (calibrated)": (scaled.max(axis=1), correct),
            },
            settings,
        ),
        report.plot_conformal_set_size(scaled, policy.conformal_qhat, settings),
        report.plot_novelty(scaled.max(axis=1), novel_conf, auroc_value, settings),
    ]


def write_smoke_kpis(results: dict[str, Any], settings: Settings) -> Path:
    # A tiny frozen snapshot the CI gate compares against, so a regression in
    # the headline numbers fails the build.
    path = Path("tests/fixtures/kpi_smoke.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "queue_macro_f1": results["queue"]["macro_f1"],
        "priority_macro_f1": results["priority"]["macro_f1"],
    }
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return path
