from __future__ import annotations

from typing import Any, Protocol

from ticketintel.logging import get_logger

logger = get_logger(__name__)

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
ENTAILMENT = "entailment"


class EntailmentScorer(Protocol):
    """Scores whether a premise entails a hypothesis.

    A protocol so unit tests can inject a deterministic stub and only the
    ml-marked tests load the real cross-encoder.
    """

    def predict(self, pairs: list[tuple[str, str]]) -> list[str]: ...


class CrossEncoderScorer:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL):
        self.model_name = model_name
        self._model: Any = None

    def _lazy_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def predict(self, pairs: list[tuple[str, str]]) -> list[str]:
        if not pairs:
            return []
        model = self._lazy_model()
        scores = model.predict(pairs)
        # The model emits (contradiction, entailment, neutral) logits; the label
        # mapping comes from the model config so it stays correct if it changes.
        label_map = getattr(model.config, "id2label", None) or {
            0: "contradiction",
            1: "entailment",
            2: "neutral",
        }
        return [str(label_map[int(row.argmax())]).lower() for row in scores]


def entailed_fraction(
    scorer: EntailmentScorer, premise: str, hypotheses: list[str]
) -> tuple[float, list[bool]]:
    # Each summary sentence is checked against the whole ticket as premise.
    if not hypotheses:
        return 1.0, []
    labels = scorer.predict([(premise, h) for h in hypotheses])
    flags = [label == ENTAILMENT for label in labels]
    return sum(flags) / len(flags), flags
