from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from ticketintel.logging import get_logger

logger = get_logger(__name__)

# Candidate inverse-regularization values searched on validation (section 3.3).
# The grid reaches past where macro-F1 plateaus so the choice is not pinned to a
# boundary.
DEFAULT_C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def _build_pipeline(c: float) -> Pipeline:
    # Word and character n-grams together. The final ablation measured the char
    # branch as very slightly negative on this data (word-only 0.5833 against
    # 0.5780 with char, reports/results.md), well inside the bootstrap interval.
    # It is kept because character grams absorb typos and product tokens, which
    # real tickets carry more of than this synthetic corpus; dropping it on a
    # 0.005 difference would be fitting the decision to noise. TF-IDF lowercases
    # internally, which is the only place the text is lowercased.
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
        ]
    )
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=0)
    return Pipeline([("features", features), ("clf", classifier)])


class TfidfClassifier:
    def __init__(self, c: float = 1.0):
        self.c = c
        self._pipeline: Pipeline | None = None

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self._fitted().named_steps["clf"].classes_)

    def _fitted(self) -> Pipeline:
        if self._pipeline is None:
            raise RuntimeError("model is not fitted")
        return self._pipeline

    def fit(self, texts: list[str], labels: list[str]) -> TfidfClassifier:
        self._pipeline = _build_pipeline(self.c).fit(texts, labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._fitted().predict_proba(texts))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"c": self.c, "pipeline": self._fitted()}, path)

    @classmethod
    def load(cls, path: Path) -> TfidfClassifier:
        payload = joblib.load(path)
        model = cls(c=payload["c"])
        model._pipeline = payload["pipeline"]
        return model


def select_c(
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
) -> tuple[TfidfClassifier, dict[float, float]]:
    # Pick C by macro-F1 on validation; hyperparameters never touch test.
    scores: dict[float, float] = {}
    best_model: TfidfClassifier | None = None
    best_score = -1.0
    for c in c_grid:
        model = TfidfClassifier(c=c).fit(train_texts, train_labels)
        pred = model.classes_[model.predict_proba(val_texts).argmax(axis=1)]
        score = float(f1_score(val_labels, pred, average="macro"))
        scores[c] = score
        if score > best_score:
            best_score, best_model = score, model
    assert best_model is not None
    logger.info("selected C", extra={"scores": scores, "best_c": best_model.c})
    return best_model, scores
