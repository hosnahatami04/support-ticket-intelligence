from __future__ import annotations

import pytest

pytestmark = pytest.mark.data
pd = pytest.importorskip("pandas")

from ticketintel.config import Settings  # noqa: E402
from ticketintel.data import schema  # noqa: E402
from ticketintel.data.split import NOVEL, split  # noqa: E402


def _settings() -> Settings:
    return Settings(
        seed=42,
        novel_queue="Outages",
        dataset={"id": "x", "revision": "y", "file": "f.csv"},
    )


def _synthetic_frame(n_per_queue: int = 200) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gid = 0
    for queue in ("Billing", "IT Support", "Sales", "Outages"):
        for i in range(n_per_queue):
            # Two rows per group so group-awareness is exercised.
            gid += 1
            for _ in range(2):
                rows.append(
                    {
                        schema.TEXT: f"{queue} ticket {i}",
                        schema.QUEUE: queue,
                        schema.PRIORITY: "low",
                        schema.GROUP_ID: gid,
                    }
                )
    return pd.DataFrame(rows)


def test_groups_never_straddle_splits() -> None:
    out = split(_synthetic_frame(), _settings())
    per_group_splits = out.groupby(schema.GROUP_ID)[schema.SPLIT].nunique()
    assert (per_group_splits == 1).all()


def test_novel_queue_only_in_holdout() -> None:
    out = split(_synthetic_frame(), _settings())
    novel_rows = out.loc[out[schema.QUEUE] == "Outages", schema.SPLIT]
    assert (novel_rows == NOVEL).all()
    assert NOVEL not in out.loc[out[schema.QUEUE] != "Outages", schema.SPLIT].to_numpy()


def test_split_ratios_within_tolerance() -> None:
    out = split(_synthetic_frame(), _settings())
    trainable = out.loc[out[schema.SPLIT] != NOVEL]
    shares = trainable[schema.SPLIT].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.02)
    assert shares["validation"] == pytest.approx(0.10, abs=0.02)
    assert shares["calibration"] == pytest.approx(0.10, abs=0.02)
    assert shares["test"] == pytest.approx(0.10, abs=0.02)


def test_split_is_deterministic() -> None:
    frame = _synthetic_frame()
    first = split(frame, _settings())[schema.SPLIT].tolist()
    second = split(frame, _settings())[schema.SPLIT].tolist()
    assert first == second


def test_no_group_leaks_between_train_and_test() -> None:
    out = split(_synthetic_frame(), _settings())
    train_groups = set(out.loc[out[schema.SPLIT] == "train", schema.GROUP_ID])
    test_groups = set(out.loc[out[schema.SPLIT] == "test", schema.GROUP_ID])
    assert train_groups.isdisjoint(test_groups)


def test_cross_queue_group_stays_in_one_split() -> None:
    # A near-duplicate group whose two rows carry different queue labels must
    # still land wholly in one split, and a group touching the novel queue must
    # go entirely to the holdout (never into training).
    frame = pd.DataFrame(
        [
            {schema.TEXT: "a", schema.QUEUE: "Billing", schema.PRIORITY: "low", schema.GROUP_ID: 1},
            {
                schema.TEXT: "a2",
                schema.QUEUE: "IT Support",
                schema.PRIORITY: "low",
                schema.GROUP_ID: 1,
            },
            {schema.TEXT: "b", schema.QUEUE: "Billing", schema.PRIORITY: "low", schema.GROUP_ID: 2},
            {
                schema.TEXT: "b2",
                schema.QUEUE: "Outages",
                schema.PRIORITY: "low",
                schema.GROUP_ID: 2,
            },
        ]
    )
    out = split(frame, _settings())
    assert out.loc[out[schema.GROUP_ID] == 1, schema.SPLIT].nunique() == 1
    # Group 2 contains an Outages (novel) row, so the whole group is held out.
    assert (out.loc[out[schema.GROUP_ID] == 2, schema.SPLIT] == NOVEL).all()
    # No novel-queue ticket ends up anywhere but the holdout.
    assert (out.loc[out[schema.QUEUE] == "Outages", schema.SPLIT] == NOVEL).all()


def test_queue_with_too_few_groups_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    # A queue small enough to be packed entirely into train leaves calibration
    # and test unrepresented; that must be surfaced, not swallowed.
    import logging

    frame = pd.DataFrame(
        [
            {schema.TEXT: "a", schema.QUEUE: "Tiny", schema.PRIORITY: "low", schema.GROUP_ID: 1},
            {schema.TEXT: "b", schema.QUEUE: "Tiny", schema.PRIORITY: "low", schema.GROUP_ID: 2},
        ]
    )
    with caplog.at_level(logging.WARNING, logger="ticketintel.data.split"):
        split(frame, _settings())

    assert any("missing from a split" in r.getMessage() for r in caplog.records)
