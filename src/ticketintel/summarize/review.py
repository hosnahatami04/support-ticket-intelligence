from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_YES = {"y", "yes", "1", "true", "t"}
_NO = {"n", "no", "0", "false", "f"}


class ReviewError(ValueError):
    pass


@dataclass(frozen=True)
class NliAccuracy:
    """How often the NLI model agreed with the human reviewer."""

    reviewed: int
    agreements: int
    false_entailments: int
    false_non_entailments: int

    @property
    def accuracy(self) -> float:
        return self.agreements / self.reviewed if self.reviewed else 0.0


def _parse_verdict(raw: str) -> bool | None:
    value = raw.strip().lower()
    if value in _YES:
        return True
    if value in _NO:
        return False
    if not value:
        return None
    raise ReviewError(f"verdict {raw!r} is neither yes nor no")


def score_review(path: Path) -> NliAccuracy:
    # Compares the reviewer's yes/no against the model's label. Rows left blank
    # are skipped, so a partially completed review still scores.
    reviewed = agreements = false_entail = false_non_entail = 0

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            human = _parse_verdict(row.get("your_verdict_yes_or_no", ""))
            if human is None:
                continue
            model_said_entailed = row["model_said"].strip() == "entailed"
            reviewed += 1
            if human == model_said_entailed:
                agreements += 1
            elif model_said_entailed:
                false_entail += 1  # model claimed support the human did not see
            else:
                false_non_entail += 1  # model missed support the human saw

    if reviewed == 0:
        raise ReviewError(f"no verdicts filled in at {path}")
    return NliAccuracy(reviewed, agreements, false_entail, false_non_entail)
