from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")

from ticketintel.reliability import conformal  # noqa: E402


def _synthetic_probs(rng: np.ndarray, n: int, n_classes: int = 4):
    # Draw calibrated-ish probability vectors and sample the true label from each
    # row's own distribution, so the model is well specified by construction.
    logits = rng.normal(size=(n, n_classes))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = exp / exp.sum(axis=1, keepdims=True)
    y_idx = np.array([rng.choice(n_classes, p=row) for row in proba])
    return proba, y_idx


def test_split_conformal_marginal_coverage_holds() -> None:
    # Split conformal guarantees marginal coverage >= 1 - alpha. Over many trials
    # the empirical coverage on fresh test data should sit near that level.
    alpha = 0.10
    rng = np.random.default_rng(0)
    coverages = []
    for _ in range(40):
        cal_proba, cal_y = _synthetic_probs(rng, 500)
        test_proba, test_y = _synthetic_probs(rng, 500)
        qhat = conformal.conformal_quantile(conformal.lac_scores(cal_proba, cal_y), alpha)
        coverages.append(conformal.empirical_coverage(test_proba, test_y, qhat))

    mean_coverage = float(np.mean(coverages))
    # The guarantee is a floor at 1 - alpha; allow a small finite-sample slack
    # below and the usual overcoverage above.
    assert mean_coverage >= (1 - alpha) - 0.03
    assert mean_coverage <= 1.0
