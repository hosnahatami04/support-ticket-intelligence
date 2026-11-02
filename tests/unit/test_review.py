from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ticketintel.summarize.review import ReviewError, score_review

HEADER = ["n", "claim_from_summary", "ticket_text", "your_verdict_yes_or_no", "model_said"]


def _sheet(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for i, (verdict, model) in enumerate(rows, 1):
            writer.writerow([i, "a claim", "a ticket", verdict, model])
    return path


def test_perfect_agreement_scores_one(tmp_path: Path) -> None:
    path = _sheet(tmp_path, [("yes", "entailed"), ("no", "not_entailed")])
    result = score_review(path)
    assert result.reviewed == 2
    assert result.accuracy == pytest.approx(1.0)
    assert result.false_entailments == 0
    assert result.false_non_entailments == 0


def test_disagreements_are_split_by_direction(tmp_path: Path) -> None:
    path = _sheet(
        tmp_path,
        [
            ("no", "entailed"),  # the model claimed support the reviewer rejected
            ("yes", "not_entailed"),  # the model missed support the reviewer saw
            ("yes", "entailed"),
        ],
    )
    result = score_review(path)
    assert result.reviewed == 3
    assert result.accuracy == pytest.approx(1 / 3)
    assert result.false_entailments == 1
    assert result.false_non_entailments == 1


def test_blank_rows_are_skipped(tmp_path: Path) -> None:
    path = _sheet(tmp_path, [("yes", "entailed"), ("", "not_entailed"), ("  ", "entailed")])
    assert score_review(path).reviewed == 1


def test_accepts_common_spellings(tmp_path: Path) -> None:
    path = _sheet(tmp_path, [("Y", "entailed"), ("TRUE", "entailed"), ("0", "not_entailed")])
    assert score_review(path).accuracy == pytest.approx(1.0)


def test_unparsable_verdict_is_rejected(tmp_path: Path) -> None:
    path = _sheet(tmp_path, [("maybe", "entailed")])
    with pytest.raises(ReviewError, match="neither yes nor no"):
        score_review(path)


def test_completely_blank_sheet_is_rejected(tmp_path: Path) -> None:
    path = _sheet(tmp_path, [("", "entailed")])
    with pytest.raises(ReviewError, match="no verdicts"):
        score_review(path)
