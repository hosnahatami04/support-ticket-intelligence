from __future__ import annotations

import numpy as np


def risk_coverage_curve(
    confidences: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Sort by confidence descending; accepting the most confident first, report
    # the running error rate (risk) at each coverage level. Coverage 1.0 is the
    # overall error rate.
    order = np.argsort(-confidences)
    correct_sorted = correct[order].astype(float)
    n = len(correct_sorted)
    cum_correct = np.cumsum(correct_sorted)
    accepted = np.arange(1, n + 1)
    coverage = accepted / n
    risk = 1.0 - cum_correct / accepted
    return coverage, risk


def area_under_risk_coverage(confidences: np.ndarray, correct: np.ndarray) -> float:
    coverage, risk = risk_coverage_curve(confidences, correct)
    return float(np.trapezoid(risk, coverage))


def select_threshold_for_risk(
    confidences: np.ndarray, correct: np.ndarray, target_risk: float
) -> tuple[float, float]:
    # Lowest confidence threshold whose accepted set keeps error <= target_risk,
    # chosen to maximise coverage. Returns (threshold, coverage_at_threshold).
    order = np.argsort(-confidences)
    conf_sorted = confidences[order]
    correct_sorted = correct[order].astype(float)
    n = len(correct_sorted)
    cum_correct = np.cumsum(correct_sorted)
    accepted = np.arange(1, n + 1)
    risk = 1.0 - cum_correct / accepted

    feasible = np.where(risk <= target_risk)[0]
    if len(feasible) == 0:
        # No threshold meets the target; abstain on everything (coverage 0).
        return float(conf_sorted[0]) + 1.0, 0.0
    k = int(feasible.max())  # accept as many as possible while staying feasible
    threshold = float(conf_sorted[k])
    coverage = (k + 1) / n
    return threshold, coverage
