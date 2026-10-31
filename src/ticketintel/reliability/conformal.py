from __future__ import annotations

import numpy as np


def lac_scores(proba: np.ndarray, y_true_idx: np.ndarray) -> np.ndarray:
    # LAC nonconformity score: 1 - p(true class). A larger score means the model
    # was less confident in the correct label.
    return np.asarray(1.0 - proba[np.arange(len(proba)), y_true_idx])


def conformal_quantile(cal_scores: np.ndarray, alpha: float = 0.10) -> float:
    # Finite-sample corrected quantile: ceil((n+1)(1-alpha)) / n. This is the
    # threshold that guarantees marginal coverage >= 1 - alpha.
    n = len(cal_scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(cal_scores, level, method="higher"))


def prediction_sets(proba: np.ndarray, qhat: float) -> list[np.ndarray]:
    # A label is in the set when its nonconformity (1 - p) is at or below qhat,
    # i.e. p >= 1 - qhat. Returns the included class indices per row.
    keep = proba >= (1.0 - qhat)
    return [np.where(row)[0] for row in keep]


def empirical_coverage(proba: np.ndarray, y_true_idx: np.ndarray, qhat: float) -> float:
    sets = prediction_sets(proba, qhat)
    hits = sum(int(y in s) for y, s in zip(y_true_idx, sets, strict=True))
    return hits / len(y_true_idx)


def average_set_size(proba: np.ndarray, qhat: float) -> float:
    sets = prediction_sets(proba, qhat)
    return float(np.mean([len(s) for s in sets]))
