from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.data

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from ticketintel.config import Settings  # noqa: E402
from ticketintel.models.registry import AVAILABLE_MODELS, build_model  # noqa: E402
from ticketintel.models.tfidf import TfidfClassifier  # noqa: E402


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        novel_queue="Spam",
        dataset={"id": "a", "revision": "b", "file": "c.csv"},
        paths={"artifacts": tmp_path},
    )


def test_builds_the_tfidf_tier(tmp_path: Path) -> None:
    assert isinstance(build_model("tfidf", _settings(tmp_path)), TfidfClassifier)


def test_unknown_model_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("nope", _settings(tmp_path))


def test_every_advertised_model_can_be_built(tmp_path: Path) -> None:
    # The registry must not advertise a tier it cannot construct.
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    for name in AVAILABLE_MODELS:
        assert build_model(name, _settings(tmp_path)) is not None
