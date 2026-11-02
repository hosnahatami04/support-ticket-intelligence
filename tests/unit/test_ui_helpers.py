from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

pytest.importorskip("streamlit")

from ticketintel.ui.app import highlight_unsupported  # noqa: E402


def test_unsupported_entities_are_marked() -> None:
    out = highlight_unsupported("Refund of 9999 euros", ["9999"])
    assert "**:red[9999]**" in out


def test_text_is_unchanged_when_everything_is_grounded() -> None:
    assert highlight_unsupported("Refund of 1200 euros", []) == "Refund of 1200 euros"
