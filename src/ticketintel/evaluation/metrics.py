from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np

from ticketintel.data.schema import PRIORITY_ORDER

# Metrics accept plain label sequences or the numpy arrays the bootstrap draws.
Labels: TypeAlias = Sequence[str] | np.ndarray


def macro_f1(y_true: Labels, y_pred: Labels) -> float:
    return float(np.mean(list(per_class_f1(y_true, y_pred).values())))


def per_class_f1(y_true: Labels, y_pred: Labels) -> dict[str, float]:
    labels = sorted(set(y_true) | set(y_pred))
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    scores: dict[str, float] = {}
    for label in labels:
        tp = int(np.sum((yp == label) & (yt == label)))
        fp = int(np.sum((yp == label) & (yt != label)))
        fn = int(np.sum((yp != label) & (yt == label)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores[label] = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return scores


def per_class_recall(y_true: Labels, y_pred: Labels) -> dict[str, float]:
    labels = sorted(set(y_true))
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    scores: dict[str, float] = {}
    for label in labels:
        support = int(np.sum(yt == label))
        tp = int(np.sum((yp == label) & (yt == label)))
        scores[label] = tp / support if support else 0.0
    return scores


def min_class_recall(y_true: Labels, y_pred: Labels) -> tuple[str, float]:
    recalls = per_class_recall(y_true, y_pred)
    worst = min(recalls, key=lambda k: recalls[k])
    return worst, recalls[worst]


def confusion_matrix(y_true: Labels, y_pred: Labels) -> tuple[list[str], np.ndarray]:
    labels = sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[index[t], index[p]] += 1
    return labels, matrix


def ordinal_mae(y_true: Labels, y_pred: Labels, order: Sequence[str] = PRIORITY_ORDER) -> float:
    rank = {label: i for i, label in enumerate(order)}
    diffs = [abs(rank[t] - rank[p]) for t, p in zip(y_true, y_pred, strict=True)]
    return float(np.mean(diffs))


def bootstrap_ci(
    y_true: Labels,
    y_pred: Labels,
    metric: Callable[[Labels, Labels], float],
    n_samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    # Percentile bootstrap: resample indices with replacement, recompute the
    # metric, return point estimate and the (alpha/2, 1-alpha/2) interval.
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    point = metric(yt, yp)
    draws = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        draws[i] = metric(yt[idx], yp[idx])
    lower = float(np.percentile(draws, 100 * alpha / 2))
    upper = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return point, lower, upper
