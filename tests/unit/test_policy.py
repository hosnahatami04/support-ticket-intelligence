from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")

from ticketintel.reliability.policy import AUTO_ROUTE, HUMAN_REVIEW, decide  # noqa: E402

_CLASSES = np.array(["billing", "it", "sales"])


def test_confident_and_singleton_routes_automatically() -> None:
    proba = np.array([0.92, 0.05, 0.03])
    # qhat small so only the top class is in the conformal set.
    d = decide(proba, _CLASSES, threshold=0.8, conformal_qhat=0.2)
    assert d.action == AUTO_ROUTE
    assert d.queue == "billing"
    assert d.candidates == ["billing"]


def test_low_confidence_goes_to_human() -> None:
    proba = np.array([0.5, 0.3, 0.2])
    d = decide(proba, _CLASSES, threshold=0.8, conformal_qhat=0.2)
    assert d.action == HUMAN_REVIEW
    assert "threshold" in d.reason


def test_ambiguous_conformal_set_goes_to_human_with_candidates() -> None:
    proba = np.array([0.55, 0.45, 0.0])
    # Large qhat keeps both plausible classes in the set, so it is ambiguous.
    d = decide(proba, _CLASSES, threshold=0.5, conformal_qhat=0.7)
    assert d.action == HUMAN_REVIEW
    assert set(d.candidates) == {"billing", "it"}
    assert "ambiguous" in d.reason
