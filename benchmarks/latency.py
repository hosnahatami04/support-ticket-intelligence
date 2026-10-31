from __future__ import annotations

from pathlib import Path

from ticketintel.config import load_settings
from ticketintel.data import schema
from ticketintel.evaluation.compare import (
    _load_split,
    dir_size_mb,
    single_ticket_latency_p95,
)
from ticketintel.logging import configure_logging, get_logger
from ticketintel.models.tfidf import TfidfClassifier

logger = get_logger(__name__)

# Artifacts to benchmark, keyed by the task they serve. A missing file is
# skipped rather than treated as an error so the script runs while some models
# are still training.
TFIDF_ARTIFACTS = {
    "queue": "tfidf_queue.joblib",
    "priority": "tfidf_priority.joblib",
}


def _benchmark_tfidf(path: Path, sample_texts: list[str]) -> dict[str, object]:
    model = TfidfClassifier.load(path)
    return {
        "latency_p95_ms": single_ticket_latency_p95(model.predict_proba, sample_texts),
        "size_mb": dir_size_mb(path),
    }


def _render(rows: list[dict[str, object]]) -> str:
    header = ["task", "artifact", "latency_p95_ms", "size_mb"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["artifact"]),
                    f"{row['latency_p95_ms']:.2f}",
                    f"{row['size_mb']:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    settings = load_settings()

    val = _load_split(settings, "validation")
    # A fixed window keeps the measurement comparable across runs and models.
    sample_texts = val[schema.TEXT].tolist()[:50]

    rows: list[dict[str, object]] = []
    for task, filename in TFIDF_ARTIFACTS.items():
        path = settings.paths.artifacts / filename
        try:
            metrics = _benchmark_tfidf(path, sample_texts)
        except Exception:
            logger.info("skipping unavailable artifact", extra={"task": task, "path": str(path)})
            continue
        rows.append({"task": task, "artifact": filename, **metrics})

    if not rows:
        logger.info("no artifacts found to benchmark")
        return

    print(_render(rows))


if __name__ == "__main__":
    main()
