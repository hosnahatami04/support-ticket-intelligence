from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never displayed
import matplotlib.pyplot as plt
import numpy as np

from ticketintel.config import Settings
from ticketintel.evaluation import metrics
from ticketintel.logging import get_logger
from ticketintel.reliability import calibration, conformal, selective

logger = get_logger(__name__)

FIGURE_NAMES = (
    "confusion_matrix",
    "reliability",
    "risk_coverage",
    "conformal_set_size",
    "novel_queue",
)


def _figures_dir(settings: Settings) -> Path:
    path = settings.paths.reports / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(y_true: list[str], y_pred: list[str], settings: Settings) -> Path:
    labels, matrix = metrics.confusion_matrix(y_true, y_pred)
    # Row-normalised so the large queues do not visually swamp the rare ones.
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted queue")
    ax.set_ylabel("true queue")
    ax.set_title("Queue confusion (row-normalised)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = _figures_dir(settings) / "confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_reliability(
    raw_conf: np.ndarray, scaled_conf: np.ndarray, correct: np.ndarray, settings: Settings
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    for conf, name in ((raw_conf, "before scaling"), (scaled_conf, "after scaling")):
        centers, acc, _ = calibration.reliability_bins(conf, correct)
        mask = ~np.isnan(acc)
        ax.plot(centers[mask], acc[mask], marker="o", label=name)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability before and after temperature scaling")
    ax.legend()
    fig.tight_layout()
    path = _figures_dir(settings) / "reliability.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_risk_coverage(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], settings: Settings
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, (conf, correct) in curves.items():
        coverage, risk = selective.risk_coverage_curve(conf, correct)
        ax.plot(coverage, risk, label=name)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (error rate on accepted tickets)")
    ax.set_title("Risk versus coverage")
    ax.legend()
    fig.tight_layout()
    path = _figures_dir(settings) / "risk_coverage.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_conformal_set_size(proba: np.ndarray, qhat: float, settings: Settings) -> Path:
    sizes = [len(s) for s in conformal.prediction_sets(proba, qhat)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sizes, bins=range(0, max(sizes) + 2), align="left", rwidth=0.85, color="#4C78A8")
    ax.set_xlabel("queues in the conformal set")
    ax.set_ylabel("tickets")
    ax.set_title("Conformal prediction set size")
    fig.tight_layout()
    path = _figures_dir(settings) / "conformal_set_size.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_novelty(familiar: np.ndarray, novel: np.ndarray, auroc: float, settings: Settings) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(familiar, bins=30, alpha=0.6, density=True, label="known queues")
    ax.hist(novel, bins=30, alpha=0.6, density=True, label="held-out queue")
    ax.set_xlabel("calibrated confidence")
    ax.set_ylabel("density")
    ax.set_title(f"Novel-queue separation (AUROC {auroc:.3f})")
    ax.legend()
    fig.tight_layout()
    path = _figures_dir(settings) / "novel_queue.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list) and len(value) == 2:
        return f"[{value[0]:.4f}, {value[1]:.4f}]"
    return str(value)


def _queue_rows(queue: dict[str, Any]) -> list[str]:
    recall = queue["min_class_recall"]
    return [
        f"| macro-F1 | {_fmt(queue['macro_f1'])} {_fmt(queue['macro_f1_ci'])} | above baseline |",
        f"| lowest class recall | {_fmt(recall['recall'])} ({recall['label']}) | report |",
        f"| ECE after calibration | {_fmt(queue['ece'])} | <= 0.03 |",
        f"| auto-route error | {_fmt(queue['auto_error_rate'])} | <= 0.05 |",
        f"| auto-route coverage | {_fmt(queue['auto_coverage'])} | >= 0.60 |",
        f"| AURC | {_fmt(queue['aurc'])} | lower is better |",
        f"| conformal coverage | {_fmt(queue['conformal_coverage'])} | 0.90 +/- 0.02 |",
        f"| mean conformal set size | {_fmt(queue['conformal_mean_set_size'])} | report |",
        f"| novel-queue AUROC | {_fmt(queue['novel_queue_auroc'])} | >= 0.75 |",
        "| error if coverage were raised to "
        f"{_fmt(queue['target_coverage'])} | {_fmt(queue['risk_at_target_coverage'])} | "
        "the cost of the coverage target |",
    ]


def _ablation_rows(abl: dict[str, Any]) -> list[str]:
    split = abl["split_strategy"]
    feats = abl["features"]
    cal = abl["calibration_and_policy"]
    return [
        "| random vs group-aware split | "
        f"{_fmt(split['random_split_macro_f1'])} vs {_fmt(split['group_aware_macro_f1'])} "
        f"(inflation {_fmt(split['inflation'])}) |",
        "| word vs word+char n-grams | "
        f"{_fmt(feats['word_only'])} vs {_fmt(feats['word_plus_char'])} "
        f"(gain {_fmt(feats['char_gain'])}) |",
        "| ECE without vs with scaling | "
        f"{_fmt(cal['ece_without_scaling'])} vs {_fmt(cal['ece_with_scaling'])} |",
        "| threshold only vs + conformal, coverage | "
        f"{_fmt(cal['threshold_only_coverage'])} vs "
        f"{_fmt(cal['threshold_plus_conformal_coverage'])} |",
        "| threshold only vs + conformal, error | "
        f"{_fmt(cal['threshold_only_error'])} vs {_fmt(cal['threshold_plus_conformal_error'])} |",
    ]


def render_results_md(results: dict[str, Any]) -> str:
    queue = results["queue"]
    priority = results["priority"]

    lines: list[str] = ["# Results", ""]
    lines.append(
        "Every number here comes from a single pass over the test split with the "
        "model and the thresholds frozen beforehand. The thresholds were fitted "
        "on the calibration split only."
    )
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    for key, value in results["provenance"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.append("## Queue routing (test)")
    lines.append("")
    lines.append("| metric | value | target |")
    lines.append("| --- | --- | --- |")
    lines.extend(_queue_rows(queue))
    lines.append("")

    lines.append("## Priority (test)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| macro-F1 | {_fmt(priority['macro_f1'])} {_fmt(priority['macro_f1_ci'])} |")
    lines.append(
        f"| ordinal MAE | {_fmt(priority['ordinal_mae'])} {_fmt(priority['ordinal_mae_ci'])} |"
    )
    lines.append("")

    summ = results.get("summarization")
    if summ:
        lines.append("## Summarization (fixed 300-ticket sample)")
        lines.append("")
        lines.append("| metric | value | target |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| schema validity | {_fmt(summ['schema_valid_rate'])} | >= 0.98 |")
        lines.append(
            f"| unsupported entities | {_fmt(summ['unsupported_entity_rate'])} | <= 0.02 |"
        )
        lines.append(f"| NLI entailed | {_fmt(summ['nli_entailed_rate'])} | >= 0.90 |")
        lines.append(f"| compression | {_fmt(summ['compression_ratio'])} | report |")
        lines.append("")

    lines.append("## Ablations")
    lines.append("")
    lines.append("| ablation | result |")
    lines.append("| --- | --- |")
    lines.extend(_ablation_rows(results["ablations"]))
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for name in FIGURE_NAMES:
        lines.append(f"- `reports/figures/{name}.png`")
    lines.append("")

    lines.append("## What didn't work")
    lines.append("")
    lines.extend(results.get("what_didnt_work", []))
    lines.append("")
    return "\n".join(lines)


def write_results_md(results: dict[str, Any], settings: Settings) -> Path:
    path = settings.paths.reports / "results.md"
    path.write_text(render_results_md(results), encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}
