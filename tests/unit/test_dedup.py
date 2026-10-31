from __future__ import annotations

import pytest

pytestmark = pytest.mark.data
pd = pytest.importorskip("pandas")

from ticketintel.data import schema  # noqa: E402
from ticketintel.data.dedup import assign_groups  # noqa: E402


def _frame(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({schema.TEXT: texts})


def _groups(texts: list[str], threshold: float = 0.90) -> list[int]:
    out = assign_groups(_frame(texts), threshold)
    return out[schema.GROUP_ID].tolist()


def test_exact_duplicates_share_a_group() -> None:
    groups = _groups(["password reset fails", "password reset fails"])
    assert groups[0] == groups[1]


def test_distinct_texts_get_distinct_groups() -> None:
    groups = _groups(
        [
            "the billing invoice is wrong this month",
            "my laptop will not connect to the office wifi",
        ]
    )
    assert groups[0] != groups[1]


def test_near_duplicates_share_a_group() -> None:
    # Same ticket with one trivial edit should land above the 0.90 threshold.
    groups = _groups(
        [
            "the printer on floor three is jammed again please help",
            "the printer on floor three is jammed again please help now",
        ]
    )
    assert groups[0] == groups[1]


def test_near_duplicate_grouping_is_transitive() -> None:
    # A~B and B~C exceed the threshold while A~C sits just below it, so only
    # transitivity through B can place A, B and C in one connected component.
    stem = (
        "the nightly data export pipeline fails intermittently with a timeout "
        "and then retries three times before giving up and paging the on call engineer"
    )
    groups = _groups(
        [
            stem + " the first failure appeared on monday",
            stem + " the first failure appeared on monday and also tuesday",
            stem + " the first failure appeared on monday and also tuesday plus wednesday morning",
        ]
    )
    assert groups[0] == groups[1] == groups[2]
