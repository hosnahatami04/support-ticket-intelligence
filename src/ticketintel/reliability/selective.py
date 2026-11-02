from __future__ import annotations

import numpy as np


def _sorted_prefix_risk(
    confidences: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Shared helper: confidence-descending order, the running error rate of each
    # prefix, and a mask marking the last index of each tie group. A threshold
    # may only be taken at a tie boundary, because `confidence >= threshold`
    # admits every sample sharing that value, not just the measured prefix.
    order = np.argsort(-confidences)
    conf_sorted = confidences[order]
    correct_sorted = correct[order].astype(float)
    n = len(correct_sorted)
    accepted = np.arange(1, n + 1)
    risk = 1.0 - np.cumsum(correct_sorted) / accepted

    last_of_group = np.empty(n, dtype=bool)
    last_of_group[:-1] = conf_sorted[:-1] > conf_sorted[1:]
    last_of_group[-1] = True
    return conf_sorted, risk, last_of_group


def risk_coverage_curve(
    confidences: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # One point per distinct confidence value, so the curve describes the model
    # rather than the order tied rows happened to arrive in. Coverage 1.0 is the
    # overall error rate.
    _, risk, last_of_group = _sorted_prefix_risk(confidences, correct)
    n = len(risk)
    coverage = (np.arange(1, n + 1) / n)[last_of_group]
    return coverage, risk[last_of_group]


def area_under_risk_coverage(confidences: np.ndarray, correct: np.ndarray) -> float:
    coverage, risk = risk_coverage_curve(confidences, correct)
    return float(np.trapezoid(risk, coverage))


def select_threshold_for_risk(
    confidences: np.ndarray, correct: np.ndarray, target_risk: float
) -> tuple[float, float]:
    # Lowest confidence threshold whose accepted set keeps error <= target_risk,
    # chosen to maximise coverage. Returns (threshold, coverage_at_threshold).
    conf_sorted, risk, last_of_group = _sorted_prefix_risk(confidences, correct)
    n = len(risk)

    # Only tie-boundary indices are admissible: at any other index the returned
    # threshold would admit more samples than the risk was measured over.
    feasible = np.where((risk <= target_risk) & last_of_group)[0]
    if len(feasible) == 0:
        # No threshold meets the target; abstain on everything (coverage 0).
        return float(conf_sorted[0]) + 1.0, 0.0
    k = int(feasible.max())  # accept as many as possible while staying feasible
    threshold = float(conf_sorted[k])
    coverage = (k + 1) / n
    return threshold, coverage
