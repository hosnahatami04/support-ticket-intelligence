from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")
sk_metrics = pytest.importorskip("sklearn.metrics")

from ticketintel.evaluation import metrics  # noqa: E402


def test_macro_f1_matches_sklearn() -> None:
    y_true = ["a", "a", "b", "c", "b", "c"]
    y_pred = ["a", "b", "b", "c", "b", "a"]
    expected = sk_metrics.f1_score(y_true, y_pred, average="macro")
    assert metrics.macro_f1(y_true, y_pred) == pytest.approx(expected)


def test_per_class_f1_hand_computed() -> None:
    # Class "a": tp=1 (idx0), fp=0, fn=1 (idx1 predicted b).
    # precision = 1/1 = 1.0, recall = 1/2 = 0.5, f1 = 2*1*0.5 / 1.5 = 0.6667.
    y_true = ["a", "a", "b"]
    y_pred = ["a", "b", "b"]
    scores = metrics.per_class_f1(y_true, y_pred)
    assert scores["a"] == pytest.approx(2 / 3)


def test_min_class_recall_finds_worst() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "a", "a", "a"]  # b is never recalled
    label, recall = metrics.min_class_recall(y_true, y_pred)
    assert label == "b"
    assert recall == pytest.approx(0.0)


def test_confusion_matrix_matches_sklearn() -> None:
    y_true = ["a", "b", "a", "c"]
    y_pred = ["a", "a", "a", "c"]
    labels, matrix = metrics.confusion_matrix(y_true, y_pred)
    expected = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels)
    assert np.array_equal(matrix, expected)


def test_ordinal_mae_hand_computed() -> None:
    # low=0, medium=1, high=2. Errors: |2-0|=2, |1-1|=0, |0-1|=1 -> mean 1.0
    y_true = ["high", "medium", "low"]
    y_pred = ["low", "medium", "medium"]
    assert metrics.ordinal_mae(y_true, y_pred) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point_and_is_deterministic() -> None:
    y_true = ["a", "b", "a", "b", "a", "b"] * 5
    y_pred = ["a", "b", "a", "a", "a", "b"] * 5
    point, lo, hi = metrics.bootstrap_ci(y_true, y_pred, metrics.macro_f1, n_samples=200, seed=1)
    assert lo <= point <= hi
    again = metrics.bootstrap_ci(y_true, y_pred, metrics.macro_f1, n_samples=200, seed=1)
    assert (point, lo, hi) == again


def test_all_wrong_scores_zero() -> None:
    # Every prediction wrong, but labels disjoint so each class has zero F1.
    y_true = ["a", "a", "a"]
    y_pred = ["b", "b", "b"]
    assert metrics.macro_f1(y_true, y_pred) == pytest.approx(0.0)


def test_ordinal_mae_zero_when_exact() -> None:
    labels = ["low", "high", "medium", "low"]
    assert metrics.ordinal_mae(labels, labels) == pytest.approx(0.0)


def test_confusion_matrix_labels_are_sorted_union() -> None:
    labels, _ = metrics.confusion_matrix(["b", "a"], ["c", "a"])
    assert labels == ["a", "b", "c"]


def test_min_class_recall_ignores_labels_only_in_pred() -> None:
    # A label that appears only in predictions is not a true class, so it must
    # not be considered when finding the worst-recalled true class.
    y_true = ["a", "a", "b"]
    y_pred = ["a", "c", "b"]  # "c" only in pred
    label, _ = metrics.min_class_recall(y_true, y_pred)
    assert label in {"a", "b"}
