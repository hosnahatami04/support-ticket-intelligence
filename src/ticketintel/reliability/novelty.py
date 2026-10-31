from __future__ import annotations

import numpy as np


def auroc(scores: np.ndarray, is_positive: np.ndarray) -> float:
    # AUROC via the rank-sum (Mann-Whitney U) identity, which needs no threshold
    # sweep. "Positive" here is the familiar (in-distribution) class, scored by
    # confidence; novel-queue tickets are the negatives and should score lower.
    pos = scores[is_positive.astype(bool)]
    neg = scores[~is_positive.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUROC needs at least one positive and one negative")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties so equal scores do not bias the statistic.
    _assign_tie_ranks(scores, order, ranks)

    rank_sum_pos = ranks[is_positive.astype(bool)].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _assign_tie_ranks(scores: np.ndarray, order: np.ndarray, ranks: np.ndarray) -> None:
    sorted_scores = scores[order]
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2  # average of ranks (i+1)..(j+1)
            ranks[order[i : j + 1]] = avg
        i = j + 1


def novel_queue_auroc(familiar_confidence: np.ndarray, novel_confidence: np.ndarray) -> float:
    scores = np.concatenate([familiar_confidence, novel_confidence])
    is_familiar = np.concatenate(
        [np.ones(len(familiar_confidence)), np.zeros(len(novel_confidence))]
    )
    return auroc(scores, is_familiar)
