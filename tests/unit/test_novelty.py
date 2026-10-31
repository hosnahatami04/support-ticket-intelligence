from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")

from ticketintel.reliability import novelty  # noqa: E402


def test_auroc_perfect_separation() -> None:
    familiar = np.array([0.9, 0.95, 0.99])
    novel = np.array([0.1, 0.2, 0.3])
    assert novelty.novel_queue_auroc(familiar, novel) == pytest.approx(1.0)


def test_auroc_random_is_half() -> None:
    familiar = np.array([0.5, 0.5])
    novel = np.array([0.5, 0.5])
    assert novelty.novel_queue_auroc(familiar, novel) == pytest.approx(0.5)


def test_auroc_matches_sklearn() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    familiar = rng.uniform(0.4, 1.0, size=50)
    novel = rng.uniform(0.0, 0.6, size=50)
    scores = np.concatenate([familiar, novel])
    labels = np.concatenate([np.ones(50), np.zeros(50)])
    assert novelty.novel_queue_auroc(familiar, novel) == pytest.approx(
        sk.roc_auc_score(labels, scores)
    )


def test_auroc_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="positive and one negative"):
        novelty.auroc(np.array([0.5, 0.6]), np.array([1, 1]))
