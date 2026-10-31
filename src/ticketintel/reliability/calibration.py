from __future__ import annotations

from itertools import pairwise

import numpy as np

# Numerical floor so log() of a zero probability stays finite.
_EPS = 1e-12


def probs_to_logits(proba: np.ndarray) -> np.ndarray:
    # Work from the probabilities every model exposes via predict_proba, so the
    # calibration layer never depends on a specific model's internal logits.
    return np.asarray(np.log(np.clip(proba, _EPS, 1.0)))


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    # Softmax(logits / T). T>1 softens the distribution, T<1 sharpens it; the
    # argmax (predicted class) is unchanged because scaling is monotonic.
    logits = probs_to_logits(proba) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True))


def negative_log_likelihood(proba: np.ndarray, y_true_idx: np.ndarray) -> float:
    picked = proba[np.arange(len(proba)), y_true_idx]
    return float(-np.mean(np.log(np.clip(picked, _EPS, 1.0))))


def fit_temperature(
    proba: np.ndarray,
    y_true_idx: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    # One-dimensional search minimising NLL on the calibration set. A manual grid
    # keeps this dependency-free and reproducible (no optimiser state).
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 1.0, 11), np.linspace(1.1, 5.0, 40)])
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = negative_log_likelihood(apply_temperature(proba, float(t)), y_true_idx)
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    # ECE over equal-width confidence bins: the support-weighted gap between mean
    # confidence and accuracy in each bin.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for lo, hi in pairwise(edges):
        in_bin = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            in_bin |= confidences == 0.0
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(confidences[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (count / n) * abs(bin_conf - bin_acc)
    return ece


def reliability_bins(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns (bin_centers, bin_accuracy, bin_confidence) for a reliability plot;
    # empty bins are reported as NaN so the caller can skip them.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = np.full(n_bins, np.nan)
    conf = np.full(n_bins, np.nan)
    for i, (lo, hi) in enumerate(pairwise(edges)):
        in_bin = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            in_bin |= confidences == 0.0
        if in_bin.any():
            acc[i] = float(correct[in_bin].mean())
            conf[i] = float(confidences[in_bin].mean())
    return centers, acc, conf
