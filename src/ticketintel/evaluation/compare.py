from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.evaluation.metrics import macro_f1
from ticketintel.logging import get_logger

logger = get_logger(__name__)

TASKS = {"queue": schema.QUEUE, "priority": schema.PRIORITY}


def _load_split(settings: Settings, split_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(settings.paths.data_processed / "tickets.parquet")
    return frame.loc[frame[schema.SPLIT] == split_name].reset_index(drop=True)


def single_ticket_latency_p95(
    predict: Callable[[list[str]], Any],
    sample_texts: list[str],
    repeats: int = 50,
    warmup: int = 5,
) -> float:
    # One ticket at a time, as the service would serve it. Report p95 in ms.
    # The warmup calls are discarded so lazy model/tokenizer initialization and
    # cold caches do not distort the timed window.
    for i in range(warmup):
        predict([sample_texts[i % len(sample_texts)]])

    timings: list[float] = []
    for i in range(repeats):
        text = sample_texts[i % len(sample_texts)]
        start = time.perf_counter()
        predict([text])
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(timings, 95))


def dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def evaluate_model_on_validation(
    model: Any, val_texts: list[str], val_labels: list[str]
) -> dict[str, float]:
    proba = model.predict_proba(val_texts)
    pred = model.classes_[proba.argmax(axis=1)]
    return {
        "macro_f1": macro_f1(val_labels, pred),
        "latency_p95_ms": single_ticket_latency_p95(model.predict_proba, val_texts[:50]),
    }


def render_comparison(rows: list[dict[str, object]]) -> str:
    # Three tiers x two tasks: macro-F1, training time, latency, model size.
    header = ["task", "model", "macro_f1", "train_seconds", "latency_p95_ms", "size_mb"]
    lines = [
        "# Model comparison\n",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["model"]),
                    f"{row['macro_f1']:.4f}",
                    f"{row['train_seconds']:.1f}",
                    f"{row['latency_p95_ms']:.2f}",
                    f"{row['size_mb']:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_comparison(rows: list[dict[str, object]], settings: Settings) -> Path:
    out_dir = settings.paths.reports / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model_comparison.md"
    path.write_text(render_comparison(rows), encoding="utf-8")
    return path
