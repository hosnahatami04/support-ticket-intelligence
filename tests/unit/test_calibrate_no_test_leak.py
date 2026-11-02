from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from ticketintel.config import Settings  # noqa: E402
from ticketintel.data import schema  # noqa: E402
from ticketintel.reliability import calibrate  # noqa: E402


def _settings() -> Settings:
    return Settings(novel_queue="Spam", dataset={"id": "a", "revision": "b", "file": "c.csv"})


def _fake_frame(split: str) -> pd.DataFrame:
    # Two clearly separable queues so a quick TF-IDF fit is confident.
    rows = []
    for i in range(20):
        rows.append(
            {
                schema.TEXT: f"invoice billing refund payment {i}",
                schema.QUEUE: "billing",
                schema.PRIORITY: "low",
                schema.GROUP_ID: i,
                schema.SPLIT: split,
            }
        )
        rows.append(
            {
                schema.TEXT: f"server crash error timeout logs {i}",
                schema.QUEUE: "tech",
                schema.PRIORITY: "high",
                schema.GROUP_ID: 100 + i,
                schema.SPLIT: split,
            }
        )
    return pd.DataFrame(rows)


def test_calibrate_never_reads_the_test_split(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    def spy_load(settings: Settings, split_name: str) -> pd.DataFrame:
        requested.append(split_name)
        return _fake_frame(split_name)

    monkeypatch.setattr(calibrate, "_load_split", spy_load)
    calibrate.calibrate_queue(_settings())

    assert "test" not in requested
    # It must have used calibration and the novel-queue probe.
    assert "calibration" in requested
    assert "novel_queue" in requested
