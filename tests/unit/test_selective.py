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


def test_threshold_admits_exactly_the_measured_set_under_ties() -> None:
    # Regression: the threshold is applied as `confidence >= tau`, so it must
    # land on a tie boundary. Taken mid-run it would admit more samples than
    # the risk was measured over and silently break the error guarantee.
    conf = np.array([0.9, 0.9, 0.9, 0.8])
    correct = np.array([1, 0, 1, 1])
    tau, coverage = selective.select_threshold_for_risk(conf, correct, target_risk=0.0)

    accepted = conf >= tau
    assert coverage == pytest.approx(accepted.mean())
    if accepted.any():
        assert (1.0 - correct[accepted].mean()) <= 0.0 + 1e-9


def test_threshold_guarantee_holds_with_saturated_confidences() -> None:
    # Many samples pile up at exactly 1.0 after scaling; the accepted set must
    # still respect the target risk.
    rng = np.random.default_rng(0)
    conf = np.concatenate([np.ones(40), rng.uniform(0.3, 0.99, 160)])
    correct = np.concatenate([rng.integers(0, 2, 40), (rng.uniform(size=160) < 0.9).astype(int)])
    tau, coverage = selective.select_threshold_for_risk(conf, correct, target_risk=0.05)

    accepted = conf >= tau
    assert coverage == pytest.approx(accepted.mean())
    if accepted.any():
        assert (1.0 - correct[accepted].mean()) <= 0.05 + 1e-9


def test_aurc_does_not_depend_on_tie_order() -> None:
    # With every confidence equal, the area must describe the model, not the
    # order the rows happened to arrive in.
    conf = np.array([0.5] * 4)
    first = selective.area_under_risk_coverage(conf, np.array([1, 1, 0, 0]))
    second = selective.area_under_risk_coverage(conf, np.array([0, 0, 1, 1]))
    assert first == pytest.approx(second)


def test_risk_coverage_curve_has_one_point_per_distinct_confidence() -> None:
    conf = np.array([0.9, 0.9, 0.5, 0.5, 0.5])
    correct = np.array([1, 0, 1, 1, 0])
    coverage, risk = selective.risk_coverage_curve(conf, correct)
    assert len(coverage) == 2  # two distinct confidence values
    assert coverage[-1] == pytest.approx(1.0)
    assert risk[-1] == pytest.approx(1 - correct.mean())
