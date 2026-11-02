from __future__ import annotations

from pathlib import Path

import pandas as pd

from ticketintel.config import Settings
from ticketintel.data import clean, dedup, load, split
from ticketintel.data import schema as sch
from ticketintel.logging import get_logger

logger = get_logger(__name__)

PROCESSED_FILENAME = "tickets.parquet"


def build(settings: Settings, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw is None:
        raw = load.load_english_tickets(settings)
    cleaned = clean.clean(raw)
    grouped = dedup.assign_groups(cleaned, settings.dedup.near_duplicate_threshold)
    return split.split(grouped, settings)


def save_processed(frame: pd.DataFrame, settings: Settings) -> Path:
    out_dir = settings.paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PROCESSED_FILENAME
    frame[[sch.TEXT, sch.QUEUE, sch.PRIORITY, sch.GROUP_ID, sch.SPLIT]].to_parquet(
        path, index=False
    )
    return path
