from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.data

pd = pytest.importorskip("pandas")
pytest.importorskip("yaml")

from ticketintel.config import Settings  # noqa: E402
from ticketintel.data import schema  # noqa: E402
from ticketintel.reliability import calibrate  # noqa: E402

_RESULTS = {
    "temperature": 1.3,
    "abstention_threshold": 0.9,
    "conformal_alpha": 0.1,
    "conformal_qhat": 0.94,
    "ece_before": 0.06,
    "ece_after": 0.03,
}


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        novel_queue="Spam",
        dataset={"id": "a", "revision": "b", "file": "c.csv"},
        paths={
            "configs": tmp_path / "configs",
            "reports": tmp_path / "reports",
            "data_processed": tmp_path / "processed",
        },
    )


def test_freeze_policy_writes_only_frozen_thresholds(tmp_path: Path) -> None:
    import yaml

    path = calibrate.freeze_policy(_RESULTS, _settings(tmp_path))
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert policy["queue"]["temperature"] == pytest.approx(1.3)
    assert policy["queue"]["abstention_threshold"] == pytest.approx(0.9)
    assert policy["queue"]["conformal_qhat"] == pytest.approx(0.94)
    # Diagnostics stay in the report, not in the frozen policy.
    assert "ece_before" not in policy["queue"]


def test_write_report_round_trips_results(tmp_path: Path) -> None:
    import json

    path = calibrate.write_report(_RESULTS, _settings(tmp_path))
    assert json.loads(path.read_text(encoding="utf-8")) == _RESULTS


def test_load_split_filters_to_the_requested_split(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.paths.data_processed.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                schema.TEXT: "a",
                schema.QUEUE: "q",
                schema.PRIORITY: "low",
                schema.GROUP_ID: 1,
                schema.SPLIT: "train",
            },
            {
                schema.TEXT: "b",
                schema.QUEUE: "q",
                schema.PRIORITY: "low",
                schema.GROUP_ID: 2,
                schema.SPLIT: "calibration",
            },
        ]
    )
    frame.to_parquet(settings.paths.data_processed / "tickets.parquet", index=False)

    out = calibrate._load_split(settings, "calibration")
    assert len(out) == 1
    assert out.loc[0, schema.TEXT] == "b"
