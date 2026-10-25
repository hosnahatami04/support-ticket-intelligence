from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.logging import get_logger

logger = get_logger(__name__)

RAW_FILENAME = "tickets_en.parquet"


def load_english_tickets(settings: Settings) -> pd.DataFrame:
    # The dataset ships as CSV files; pin the revision and pull only the one file
    # the data card settled on, then keep the English rows (section 3.2).
    ds = load_dataset(
        settings.dataset.id,
        revision=settings.dataset.revision,
        data_files=settings.dataset.file,
        split="train",
    )
    frame: pd.DataFrame = ds.to_pandas()
    language = frame[schema.LANGUAGE].astype("string").str.strip().str.lower()
    english = frame.loc[language == settings.dataset.language].reset_index(drop=True)
    logger.info(
        "loaded tickets",
        extra={"total": len(frame), "english": len(english), "file": settings.dataset.file},
    )
    return english


def save_raw(frame: pd.DataFrame, settings: Settings) -> Path:
    out_dir = settings.paths.data_raw
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RAW_FILENAME
    frame.to_parquet(path, index=False)
    return path


def run(settings: Settings) -> Path:
    return save_raw(load_english_tickets(settings), settings)
