from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from ticketintel.models.embeddings import EmbeddingCache, EmbeddingClassifier  # noqa: E402


def test_cache_round_trips_vectors(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, "model-tag")
    assert cache.get("hello") is None
    vector = np.arange(4, dtype=float)
    cache.put("hello", vector)
    assert np.array_equal(cache.get("hello"), vector)


def test_cache_key_depends_on_model_tag(tmp_path: Path) -> None:
    EmbeddingCache(tmp_path, "tag-a").put("same text", np.zeros(3))
    assert EmbeddingCache(tmp_path, "tag-b").get("same text") is None


class _CountingEncoder:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        self.encoded.extend(texts)
        return np.ones((len(texts), 3), dtype=float)


def test_embedding_cache_avoids_recomputation(tmp_path: Path) -> None:
    model = EmbeddingClassifier(tmp_path)
    encoder = _CountingEncoder()
    model._encoder = encoder  # inject a fake encoder, no model download

    model._embed(["alpha", "beta"])
    model._embed(["alpha", "gamma"])  # alpha and beta already cached

    # "alpha" is encoded once; the second call only encodes the new "gamma".
    assert encoder.encoded == ["alpha", "beta", "gamma"]
