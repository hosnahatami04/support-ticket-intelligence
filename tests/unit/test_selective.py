from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")

from ticketintel.reliability import selective  # noqa: E402


def test_coverage_one_equals_total_error_rate() -> None:
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 0, 1, 0])  # 50% error overall
    coverage, risk = selective.risk_coverage_curve(conf, correct)
    assert coverage[-1] == pytest.approx(1.0)
    assert risk[-1] == pytest.approx(0.5)


def test_risk_is_lower_at_low_coverage_when_confidence_tracks_correctness() -> None:
    # Most confident predictions are correct, so risk rises with coverage.
    conf = np.array([0.99, 0.95, 0.4, 0.3])
    correct = np.array([1, 1, 0, 0])
    _coverage, risk = selective.risk_coverage_curve(conf, correct)
    assert risk[0] == pytest.approx(0.0)
    assert risk[-1] == pytest.approx(0.5)


def test_threshold_meets_target_risk() -> None:
    conf = np.array([0.99, 0.95, 0.9, 0.5, 0.4])
    correct = np.array([1, 1, 1, 0, 0])
    tau, coverage = selective.select_threshold_for_risk(conf, correct, target_risk=0.0)
    # The three confident, correct predictions are accepted at zero risk.
    accepted = conf >= tau
    assert correct[accepted].mean() == pytest.approx(1.0)
    assert coverage == pytest.approx(0.6)


def test_aurc_hand_computed_monotone_case() -> None:
    # Perfectly ordered: risk stays 0 until the single error at full coverage.
    conf = np.array([0.9, 0.8, 0.7])
    correct = np.array([1, 1, 0])
    aurc = selective.area_under_risk_coverage(conf, correct)
    assert aurc >= 0.0
    # Worse ordering must not produce a smaller area.
    worse = selective.area_under_risk_coverage(conf, np.array([0, 1, 1]))
    assert worse >= aurc


def test_infeasible_target_abstains_entirely() -> None:
    # Even the single most confident prediction is wrong, so no threshold can
    # reach a 0% error target: the policy must abstain on everything.
    conf = np.array([0.9, 0.8])
    correct = np.array([0, 0])
    tau, coverage = selective.select_threshold_for_risk(conf, correct, target_risk=0.0)
    assert coverage == pytest.approx(0.0)
    assert (conf >= tau).sum() == 0
