from __future__ import annotations

import json
from typing import Any

import pytest

from ticketintel.summarize.client import OllamaClient, OllamaConfig, OllamaError


class _FakeResponse:
    def __init__(self, body: dict[str, Any]):
        self._body = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _client(**kw: Any) -> OllamaClient:
    return OllamaClient(OllamaConfig(model="m", timeout_s=0.1, retry_backoff_s=0.0, **kw))


def test_generate_json_parses_model_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse({"response": '{"issue": "disk full"}'}),
    )
    assert _client().generate_json("p") == {"issue": "disk full"}


def test_generate_json_rejects_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _FakeResponse({"response": "not json"})
    )
    with pytest.raises(OllamaError, match="valid JSON"):
        _client().generate_json("p")


def test_generate_json_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _FakeResponse({"response": "[1, 2]"})
    )
    with pytest.raises(OllamaError, match="not an object"):
        _client().generate_json("p")


def test_timeout_is_retried_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def boom(*a: object, **k: object) -> None:
        calls.append(1)
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(OllamaError, match="failed after 3 attempts"):
        _client(max_retries=2).generate_json("p")
    assert len(calls) == 3  # initial attempt plus two retries


def test_retry_succeeds_after_one_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"n": 0}

    def flaky(*a: object, **k: object) -> _FakeResponse:
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("transient")
        return _FakeResponse({"response": '{"issue": "ok"}'})

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    assert _client(max_retries=2).generate_json("p") == {"issue": "ok"}
    assert state["n"] == 2


def test_model_digest_found_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse({"models": [{"model": "m", "digest": "abc"}]}),
    )
    assert _client().model_digest() == "abc"

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse({"models": []}))
    with pytest.raises(OllamaError, match="not available"):
        _client().model_digest()
