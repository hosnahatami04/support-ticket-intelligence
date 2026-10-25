from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.logging import get_logger

logger = get_logger(__name__)

NOVEL = "novel_queue"
_SPLITS = ("train", "validation", "calibration", "test")


def _assign_groups_to_splits(
    group_sizes: dict[int, int], ratios: list[float], rng: np.random.Generator
) -> dict[int, str]:
    # Greedy proportional packing: largest groups first, each whole group placed
    # into the split currently furthest below its target share.
    total = sum(group_sizes.values())
    targets = {name: ratio * total for name, ratio in zip(_SPLITS, ratios, strict=True)}
    current = dict.fromkeys(_SPLITS, 0.0)
    assignment: dict[int, str] = {}

    order = list(group_sizes)
    rng.shuffle(order)
    order.sort(key=lambda g: group_sizes[g], reverse=True)

    for group in order:
        size = group_sizes[group]
        deficits = {name: targets[name] - current[name] for name in _SPLITS}
        chosen = max(_SPLITS, key=lambda name: (deficits[name], name))
        assignment[group] = chosen
        current[chosen] += size
    return assignment


def _group_table(frame: pd.DataFrame, novel_queue: str) -> pd.DataFrame:
    # One row per duplicate group, carrying its size, its dominant queue, and
    # whether any member belongs to the held-out queue. Groups are the atomic
    # unit so a near-duplicate pair can never straddle splits, even when its
    # rows carry different queue labels.
    rows = []
    for group_id, rows_in_group in frame.groupby(schema.GROUP_ID, sort=True):
        queue = rows_in_group[schema.QUEUE].mode().iloc[0]
        has_novel = bool((rows_in_group[schema.QUEUE] == novel_queue).any())
        rows.append((int(cast(int, group_id)), str(queue), len(rows_in_group), has_novel))
    return pd.DataFrame(rows, columns=["group_id", "queue", "size", "has_novel"])


def split(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    out = frame.copy()
    groups = _group_table(out, settings.novel_queue)

    group_split: dict[int, str] = {}

    # Any group containing a held-out-queue ticket goes entirely to the probe so
    # the model never trains on that queue, even via a near-duplicate (3.2).
    novel_groups = groups.loc[groups["has_novel"], "group_id"]
    for gid in novel_groups:
        group_split[int(gid)] = NOVEL

    trainable = groups.loc[~groups["has_novel"]]
    ratios = [
        settings.split.train,
        settings.split.validation,
        settings.split.calibration,
        settings.split.test,
    ]
    rng = np.random.default_rng(settings.seed)

    # Stratify by dominant queue: assign whole groups within each queue so the
    # per-queue proportions hold.
    for _, queue_groups in trainable.groupby("queue", sort=True):
        sizes = {
            int(cast(int, r.group_id)): int(cast(int, r.size)) for r in queue_groups.itertuples()
        }
        group_split.update(_assign_groups_to_splits(sizes, ratios, rng))

    out[schema.SPLIT] = out[schema.GROUP_ID].map(group_split)

    split_counts = out[schema.SPLIT].value_counts()
    logger.info(
        "assigned splits",
        extra={
            str(k): int(v)
            for k, v in zip(split_counts.index.to_numpy(), split_counts.to_numpy(), strict=True)
        },
    )
    return out
