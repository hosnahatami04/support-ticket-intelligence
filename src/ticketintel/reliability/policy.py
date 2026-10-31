from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AUTO_ROUTE = "auto_route"
HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class Decision:
    action: str
    queue: str
    confidence: float
    candidates: list[str]
    reason: str


def decide(
    proba_row: np.ndarray,
    classes: np.ndarray,
    threshold: float,
    conformal_qhat: float,
) -> Decision:
    # Auto-route only when the model is both confident (calibrated max prob at or
    # above the abstention threshold) and unambiguous (the conformal set holds a
    # single queue). Anything else goes to a human with the candidate queues.
    top_idx = int(proba_row.argmax())
    confidence = float(proba_row[top_idx])
    predicted = str(classes[top_idx])

    in_set = proba_row >= (1.0 - conformal_qhat)
    candidates = [str(c) for c in classes[in_set]]
    if not candidates:  # degenerate: fall back to the single top class
        candidates = [predicted]

    if confidence >= threshold and len(candidates) == 1:
        return Decision(AUTO_ROUTE, predicted, confidence, candidates, "confident and unambiguous")

    if confidence < threshold:
        reason = "confidence below abstention threshold"
    else:
        reason = "conformal set is ambiguous"
    return Decision(HUMAN_REVIEW, predicted, confidence, candidates, reason)
