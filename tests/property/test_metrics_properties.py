from __future__ import annotations

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from ticketintel.data.schema import PRIORITY_ORDER  # noqa: E402
from ticketintel.evaluation import metrics  # noqa: E402

_QUEUE_LABELS = ["billing", "tech", "sales", "hr"]
_labels = st.lists(st.sampled_from(_QUEUE_LABELS), min_size=1, max_size=60)
_priorities = st.lists(st.sampled_from(PRIORITY_ORDER), min_size=1, max_size=60)


@st.composite
def _paired(draw: st.DrawFn, pool: list[str]) -> tuple[list[str], list[str]]:
    # Equal-length true/pred label sequences drawn from the same pool.
    n = draw(st.integers(min_value=1, max_value=60))
    choices = st.lists(st.sampled_from(pool), min_size=n, max_size=n)
    return draw(choices), draw(choices)


@given(_paired(_QUEUE_LABELS))
def test_macro_f1_in_unit_interval(pair: tuple[list[str], list[str]]) -> None:
    y_true, y_pred = pair
    assert 0.0 <= metrics.macro_f1(y_true, y_pred) <= 1.0


@given(_labels)
def test_perfect_prediction_scores_one(y_true: list[str]) -> None:
    assert metrics.macro_f1(y_true, y_true) == pytest.approx(1.0)
    _, worst_recall = metrics.min_class_recall(y_true, y_true)
    assert worst_recall == pytest.approx(1.0)


@given(_paired(_QUEUE_LABELS))
def test_recall_in_unit_interval(pair: tuple[list[str], list[str]]) -> None:
    y_true, y_pred = pair
    for recall in metrics.per_class_recall(y_true, y_pred).values():
        assert 0.0 <= recall <= 1.0


@given(_paired(list(PRIORITY_ORDER)))
def test_ordinal_mae_is_symmetric_and_nonnegative(
    pair: tuple[list[str], list[str]],
) -> None:
    y_true, y_pred = pair
    forward = metrics.ordinal_mae(y_true, y_pred)
    backward = metrics.ordinal_mae(y_pred, y_true)
    assert forward >= 0.0
    assert forward == pytest.approx(backward)


@given(_paired(_QUEUE_LABELS))
def test_confusion_matrix_total_equals_sample_count(
    pair: tuple[list[str], list[str]],
) -> None:
    y_true, y_pred = pair
    _, matrix = metrics.confusion_matrix(y_true, y_pred)
    assert int(matrix.sum()) == len(y_true)


@given(_paired(_QUEUE_LABELS))
def test_bootstrap_interval_is_ordered_and_bounded(
    pair: tuple[list[str], list[str]],
) -> None:
    # The percentile interval can, for skewed small samples, exclude the point
    # estimate, so the invariant to hold always is: bounds are ordered and the
    # metric (macro-F1) stays in [0, 1]. Reproducibility is covered by a unit
    # test with a fixed seed.
    y_true, y_pred = pair
    point, lo, hi = metrics.bootstrap_ci(y_true, y_pred, metrics.macro_f1, n_samples=100, seed=7)
    assert lo <= hi
    assert 0.0 <= point <= 1.0
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
