from __future__ import annotations

from ticketintel.config import Settings
from ticketintel.models.base import TicketClassifier
from ticketintel.models.tfidf import TfidfClassifier


def build_model(name: str, settings: Settings) -> TicketClassifier:
    # Factory keyed by model tier. The embedding and transformer tiers import
    # their heavy dependencies lazily so building a tfidf model stays light.
    if name == "tfidf":
        return TfidfClassifier()
    if name == "embeddings":
        from ticketintel.models.embeddings import EmbeddingClassifier

        return EmbeddingClassifier(settings.paths.artifacts / "embedding_cache")
    if name == "transformer":
        from ticketintel.models.transformer import TransformerClassifier

        return TransformerClassifier()
    raise ValueError(f"unknown model: {name!r}")


AVAILABLE_MODELS = ("tfidf", "embeddings", "transformer")
