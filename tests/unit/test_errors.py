from __future__ import annotations

from ticketintel.evaluation.errors import top_confusions


def test_top_confusions_ignores_correct_and_ranks_by_count() -> None:
    y_true = ["a", "a", "a", "b", "c"]
    y_pred = ["b", "b", "a", "c", "c"]  # a->b twice, b->c once, c correct once
    sample_ids = [10, 11, 12, 13, 14]
    pairs = top_confusions(y_true, y_pred, sample_ids)

    assert pairs[0].true_label == "a"
    assert pairs[0].pred_label == "b"
    assert pairs[0].count == 2
    assert pairs[0].sample_ids == [10, 11]
    # The correctly classified "c" never appears as a confusion.
    assert all(p.true_label != p.pred_label for p in pairs)


def test_top_confusions_caps_sample_ids() -> None:
    y_true = ["a"] * 8
    y_pred = ["b"] * 8
    pairs = top_confusions(y_true, y_pred, list(range(8)), max_ids=3)
    assert pairs[0].count == 8
    assert len(pairs[0].sample_ids) == 3
