from __future__ import annotations

import pytest

pytestmark = pytest.mark.data
pd = pytest.importorskip("pandas")

from ticketintel.data import schema  # noqa: E402
from ticketintel.data.clean import SchemaError, clean  # noqa: E402


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(schema.SOURCE_COLUMNS))


def test_clean_builds_text_and_strips_html() -> None:
    frame = _frame(
        [
            {
                "subject": "Login  broken",
                "body": "<p>Cannot sign in</p>   after   reset",
                "queue": "IT Support",
                "priority": "High",
                "language": "en",
            }
        ]
    )
    out = clean(frame)
    assert out.loc[0, schema.TEXT] == "Login broken\nCannot sign in after reset"
    assert out.loc[0, schema.PRIORITY] == "high"


def test_clean_drops_rows_without_text() -> None:
    frame = _frame(
        [
            {"subject": "", "body": "   ", "queue": "Billing", "priority": "low", "language": "en"},
            {
                "subject": "Ok",
                "body": "real",
                "queue": "Billing",
                "priority": "low",
                "language": "en",
            },
        ]
    )
    out = clean(frame)
    assert len(out) == 1


def test_clean_drops_rows_with_unknown_priority() -> None:
    frame = _frame(
        [
            {
                "subject": "a",
                "body": "b",
                "queue": "Billing",
                "priority": "urgent",
                "language": "en",
            },
        ]
    )
    assert clean(frame).empty


def test_clean_requires_source_columns() -> None:
    with pytest.raises(SchemaError, match="missing required columns"):
        clean(pd.DataFrame({"subject": ["x"]}))
