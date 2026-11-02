from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from ticketintel.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingCache:
    """Disk cache of sentence embeddings keyed by (model tag, text hash).

    Embeddings are deterministic for a fixed model, so caching by content hash
    lets evaluation reuse vectors across runs instead of recomputing them.
    """

    def __init__(self, cache_dir: Path, model_tag: str):
        self.cache_dir = cache_dir
        self.model_tag = model_tag
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str) -> Path:
        digest = hashlib.sha1(f"{self.model_tag}\x00{text}".encode()).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def get(self, text: str) -> np.ndarray | None:
        path = self._path(text)
        if path.exists():
            return np.asarray(np.load(path))
        return None

    def put(self, text: str, vector: np.ndarray) -> None:
        np.save(self._path(text), vector)


class EmbeddingClassifier:
    def __init__(self, cache_dir: Path, model_name: str = DEFAULT_MODEL, c: float = 1.0):
        self.model_name = model_name
        self.c = c
        self.cache_dir = cache_dir
        self._cache = EmbeddingCache(cache_dir, model_name)
        self._encoder: SentenceTransformer | None = None
        self._clf: LogisticRegression | None = None

    def _lazy_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name, device="cpu")
        return self._encoder

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray | None] = [self._cache.get(t) for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            fresh = self._lazy_encoder().encode(
                [texts[i] for i in missing], convert_to_numpy=True, normalize_embeddings=True
            )
            for position, vector in zip(missing, fresh, strict=True):
                self._cache.put(texts[position], vector)
                vectors[position] = vector
        logger.info("embedded texts", extra={"total": len(texts), "cache_misses": len(missing)})
        return np.vstack([v for v in vectors if v is not None])

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self._fitted().classes_)

    def _fitted(self) -> LogisticRegression:
        if self._clf is None:
            raise RuntimeError("model is not fitted")
        return self._clf

    def fit(self, texts: list[str], labels: list[str]) -> EmbeddingClassifier:
        features = self._embed(texts)
        clf = LogisticRegression(C=self.c, class_weight="balanced", max_iter=2000, random_state=0)
        clf.fit(features, labels)
        self._clf = clf
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._fitted().predict_proba(self._embed(texts)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "c": self.c,
            "cache_dir": str(self.cache_dir),
            "clf": self._fitted(),
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> EmbeddingClassifier:
        payload = joblib.load(path)
        model = cls(Path(payload["cache_dir"]), model_name=payload["model_name"], c=payload["c"])
        model._clf = payload["clf"]
        return model
