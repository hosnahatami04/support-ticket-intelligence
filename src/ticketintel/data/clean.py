from __future__ import annotations

import re

import pandas as pd

from ticketintel.data import schema
from ticketintel.logging import get_logger

logger = get_logger(__name__)

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class SchemaError(ValueError):
    pass


def _normalize(value: object) -> str:
    text = "" if value is None else str(value)
    text = _HTML_TAG.sub(" ", text)
    # Collapse runs of spaces/tabs and excessive blank lines, but keep the
    # single newline between subject and body: casing is left untouched because
    # only TF-IDF lowercases, downstream models see the original text.
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [c for c in schema.SOURCE_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")


def build_text(frame: pd.DataFrame) -> pd.Series:
    subject = frame[schema.SUBJECT].map(_normalize)
    body = frame[schema.BODY].map(_normalize)
    return (subject + "\n" + body).str.strip()


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame)
    out = frame.copy()
    out[schema.TEXT] = build_text(out)
    out[schema.QUEUE] = out[schema.QUEUE].astype("string").str.strip()
    out[schema.PRIORITY] = out[schema.PRIORITY].astype("string").str.strip().str.lower()

    before = len(out)
    has_text = out[schema.TEXT].str.len() > 0
    has_queue = out[schema.QUEUE].str.len() > 0
    has_priority = out[schema.PRIORITY].isin(schema.PRIORITY_ORDER)
    out = out.loc[has_text & has_queue & has_priority].reset_index(drop=True)
    logger.info("cleaned tickets", extra={"before": before, "after": len(out)})

    return out[[schema.TEXT, schema.QUEUE, schema.PRIORITY]]
