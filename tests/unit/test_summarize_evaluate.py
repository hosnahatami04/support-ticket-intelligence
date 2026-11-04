from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.data

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("yaml")

from ticketintel.config import Settings  # noqa: E402
from ticketintel.data import schema  # noqa: E402
from ticketintel.summarize import evaluate as ev  # noqa: E402


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        novel_queue="Spam",
        dataset={"id": "a", "revision": "b", "file": "c.csv"},
        paths={
            "data_processed": tmp_path / "processed",
            "reports": tmp_path / "reports",
            "artifacts": tmp_path / "artifacts",
            "configs": tmp_path / "configs",
        },
    )


def _cfg(size: int = 5) -> ev.SummarizerSettings:
    return ev.SummarizerSettings(
        model="m",
        host="h",
        temperature=0.0,
        timeout_s=1.0,
        max_retries=0,
        prompt_version="summary_v1",
        eval_sample_size=size,
        eval_seed=42,
    )


def _write_tickets(settings: Settings, n: int = 20) -> None:
    settings.paths.data_processed.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                schema.TEXT: f"ticket {i}",
                schema.QUEUE: "q",
                schema.PRIORITY: "low",
                schema.GROUP_ID: i,
                schema.SPLIT: "test" if i % 2 == 0 else "train",
            }
            for i in range(n)
        ]
    )
    frame.to_parquet(settings.paths.data_processed / "tickets.parquet", index=False)


def test_sample_is_drawn_from_test_only_and_is_deterministic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_tickets(settings)
    first = ev.sample_tickets(settings, _cfg())
    second = ev.sample_tickets(settings, _cfg())
    assert list(first[schema.TEXT]) == list(second[schema.TEXT])
    assert set(first[schema.SPLIT]) == {"test"}


def test_sample_caps_at_available_rows(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_tickets(settings, n=6)  # only 3 test rows
    assert len(ev.sample_tickets(settings, _cfg(size=300))) == 3


def test_review_file_is_balanced_and_has_blank_verdicts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    rows = [
        {"sentence": f"e{i}", "nli_label": "entailed", "ticket_excerpt": "x"} for i in range(50)
    ]
    rows += [
        {"sentence": f"n{i}", "nli_label": "not_entailed", "ticket_excerpt": "x"} for i in range(50)
    ]

    path = ev.write_review_file(rows, settings, per_class=30)
    picked = json.loads(path.read_text(encoding="utf-8"))

    labels = [r["nli_label"] for r in picked]
    assert labels.count("entailed") == 30
    assert labels.count("not_entailed") == 30
    # The agent must not pre-fill a judgement.
    assert all(r["human_verdict"] == "" for r in picked)


def test_report_excludes_private_keys(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = ev.write_report({"schema_valid_rate": 1.0, "_review_rows": [1, 2]}, settings)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == {"schema_valid_rate": 1.0}


def test_failures_are_recorded_with_their_reason(tmp_path: Path) -> None:
    # Regression: the generator counted failures but discarded why, which made a
    # persistent failure impossible to diagnose from the report alone.
    settings = _settings(tmp_path)
    failures = [
        {"reason": "summary failed after 2 attempts: bad json", "ticket_excerpt": "a ticket"}
    ]

    path = ev.write_failures(failures, settings)
    written = json.loads(path.read_text(encoding="utf-8"))

    assert written == failures
    assert "bad json" in written[0]["reason"]
    assert path.name == "summarization_failures.json"


def test_latency_defaults_to_a_sample_large_enough_for_a_p95() -> None:
    # Regression: the default was 10, which makes p95 the maximum of ten draws.
    # A single unrelated stall on the host then lands in the report as a
    # measured latency; an unchanged code path was observed swinging from
    # 3.1 s to 6.3 s for that reason alone.
    import inspect

    default = inspect.signature(ev.measure_latency).parameters["n"].default
    assert default >= 20, "a p95 needs more than a handful of samples"
