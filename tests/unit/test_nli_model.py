from __future__ import annotations

import pytest

pytestmark = pytest.mark.ml

pytest.importorskip("sentence_transformers")
pytest.importorskip("torch")

from ticketintel.summarize.nli import CrossEncoderScorer, entailed_fraction  # noqa: E402


def test_real_nli_separates_entailed_from_contradicted() -> None:
    # A clear pair either way: the real model should not label a direct
    # contradiction as entailment.
    premise = "The customer reset the password three times and the portal is still down."
    scorer = CrossEncoderScorer()
    fraction, flags = entailed_fraction(
        scorer,
        premise,
        ["The portal is down.", "The portal is working normally."],
    )
    assert flags[0] is True
    assert flags[1] is False
    assert fraction == pytest.approx(0.5)
