from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")

from ticketintel.reliability import calibration  # noqa: E402


def test_ece_zero_for_perfectly_calibrated() -> None:
    # Confidence equals accuracy in every bin: 60% confident and 60% correct.
    confidences = np.array([0.6] * 10)
    correct = np.array([1] * 6 + [0] * 4)
    assert calibration.expected_calibration_error(confidences, correct) == pytest.approx(0.0)


def test_ece_hand_computed_gap() -> None:
    # All in one bin: mean confidence 0.9, accuracy 0.5 -> ECE = 0.4.
    confidences = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([1, 0, 1, 0])
    assert calibration.expected_calibration_error(confidences, correct) == pytest.approx(0.4)


def test_temperature_preserves_argmax() -> None:
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    for t in (0.5, 1.0, 2.0, 4.0):
        scaled = calibration.apply_temperature(proba, t)
        assert np.array_equal(scaled.argmax(axis=1), proba.argmax(axis=1))
        assert np.allclose(scaled.sum(axis=1), 1.0)


def test_fit_temperature_does_not_increase_nll() -> None:
    rng = np.random.default_rng(0)
    # Overconfident predictions: true class usually right but probability too high.
    proba = np.array([[0.95, 0.03, 0.02]] * 50 + [[0.02, 0.95, 0.03]] * 50)
    y_idx = np.array([0] * 45 + [1] * 5 + [1] * 45 + [0] * 5)
    rng.shuffle(y_idx)
    t = calibration.fit_temperature(proba, y_idx)
    base = calibration.negative_log_likelihood(proba, y_idx)
    tuned = calibration.negative_log_likelihood(calibration.apply_temperature(proba, t), y_idx)
    assert tuned <= base + 1e-9


def test_reliability_bins_marks_empty_as_nan() -> None:
    confidences = np.array([0.95, 0.95])
    correct = np.array([1, 0])
    _, acc, _conf = calibration.reliability_bins(confidences, correct, n_bins=5)
    # Only the top bin is populated; the rest are NaN.
    assert np.isnan(acc[:-1]).all()
    assert not np.isnan(acc[-1])
