from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")

from ticketintel.models.base import TicketClassifier  # noqa: E402
from ticketintel.models.tfidf import TfidfClassifier, select_c  # noqa: E402

# Hand-written, clearly separable synthetic tickets (never copied from the data).
_BILLING = [
    "invoice charge refund payment billing statement overcharge",
    "billing refund invoice payment double charged this month",
    "payment failed invoice refund requested billing department",
]
_TECH = [
    "server error timeout crash stack trace exception logs",
    "application crash timeout server exception error restart",
    "timeout exception server crash error reproduce logs attached",
]


def _dataset() -> tuple[list[str], list[str]]:
    texts = _BILLING + _TECH
    labels = ["billing"] * len(_BILLING) + ["tech"] * len(_TECH)
    return texts, labels


def test_tfidf_satisfies_protocol() -> None:
    # A fitted model exposes the full protocol surface. (isinstance against a
    # runtime_checkable Protocol would evaluate the classes_ property, which
    # raises before fit, so conformance is checked on a fitted instance.)
    texts, labels = _dataset()
    model = TfidfClassifier().fit(texts, labels)
    assert isinstance(model, TicketClassifier)


def test_tfidf_separates_clear_classes() -> None:
    texts, labels = _dataset()
    model = TfidfClassifier(c=10.0).fit(texts, labels)
    proba = model.predict_proba(texts)
    pred = model.classes_[proba.argmax(axis=1)]
    assert list(pred) == labels


def test_proba_rows_sum_to_one() -> None:
    texts, labels = _dataset()
    model = TfidfClassifier().fit(texts, labels)
    proba = model.predict_proba(texts)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_save_load_round_trip(tmp_path: Path) -> None:
    texts, labels = _dataset()
    model = TfidfClassifier(c=3.0).fit(texts, labels)
    before = model.predict_proba(texts)
    path = tmp_path / "tfidf.joblib"
    model.save(path)
    reloaded = TfidfClassifier.load(path)
    assert np.allclose(reloaded.predict_proba(texts), before)
    assert reloaded.c == 3.0


def test_select_c_returns_scores_for_each_candidate() -> None:
    texts, labels = _dataset()
    model, scores = select_c(texts, labels, texts, labels, c_grid=(0.3, 1.0, 10.0))
    assert set(scores) == {0.3, 1.0, 10.0}
    assert model.c in scores


def test_tfidf_is_reproducible() -> None:
    texts, labels = _dataset()
    first = TfidfClassifier(c=1.0).fit(texts, labels).predict_proba(texts)
    second = TfidfClassifier(c=1.0).fit(texts, labels).predict_proba(texts)
    assert np.array_equal(first, second)
