from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")

from ticketintel.reliability import conformal  # noqa: E402


def test_lac_score_is_one_minus_true_prob() -> None:
    proba = np.array([[0.7, 0.3], [0.2, 0.8]])
    scores = conformal.lac_scores(proba, np.array([0, 1]))
    assert np.allclose(scores, [0.3, 0.2])


def test_quantile_uses_finite_sample_correction() -> None:
    scores = np.linspace(0.0, 1.0, 100)
    qhat = conformal.conformal_quantile(scores, alpha=0.10)
    # ceil(101 * 0.9) / 100 = 0.91 quantile, >= the naive 0.90 quantile.
    assert qhat >= float(np.quantile(scores, 0.90, method="higher"))


def test_prediction_set_includes_true_label_when_covered() -> None:
    proba = np.array([[0.6, 0.4], [0.1, 0.9]])
    # qhat large enough that both plausible labels are kept for row 0.
    sets = conformal.prediction_sets(proba, qhat=0.7)
    assert 0 in sets[0]


def test_average_set_size_grows_with_qhat() -> None:
    proba = np.array([[0.5, 0.3, 0.2]] * 10)
    small = conformal.average_set_size(proba, qhat=0.3)
    large = conformal.average_set_size(proba, qhat=0.9)
    assert large >= small
