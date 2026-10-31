from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class ConfusionPair:
    true_label: str
    pred_label: str
    count: int
    sample_ids: list[int] = field(default_factory=list)


def top_confusions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    sample_ids: Sequence[int],
    top_k: int = 10,
    max_ids: int = 5,
) -> list[ConfusionPair]:
    # Report only misclassified pairs, with a few example indices rather than the
    # full ticket text, so error reports never embed dataset content.
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for t, p, sid in zip(y_true, y_pred, sample_ids, strict=True):
        if t != p:
            buckets[(t, p)].append(int(sid))

    pairs = [ConfusionPair(t, p, len(ids), ids[:max_ids]) for (t, p), ids in buckets.items()]
    pairs.sort(key=lambda c: c.count, reverse=True)
    return pairs[:top_k]
