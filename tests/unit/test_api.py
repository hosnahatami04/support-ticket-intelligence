from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("sklearn")
yaml = pytest.importorskip("yaml")

from fastapi.testclient import TestClient  # noqa: E402

from ticketintel.models.tfidf import TfidfClassifier  # noqa: E402
from ticketintel.service import api  # noqa: E402

_BILLING = ["invoice refund billing payment charge", "billing payment invoice refund due"]
_TECH = ["server crash timeout error logs", "timeout crash server exception trace"]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    texts = _BILLING + _TECH
    labels = ["billing"] * 2 + ["tech"] * 2
    queue_model = TfidfClassifier(c=10.0).fit(texts, labels)
    priority_model = TfidfClassifier(c=10.0).fit(texts, ["low", "low", "high", "high"])

    monkeypatch.setattr(api.state, "queue_model", queue_model)
    monkeypatch.setattr(api.state, "priority_model", priority_model)
    monkeypatch.setattr(
        api.state,
        "policy",
        api.Policy(temperature=1.0, abstention_threshold=0.5, conformal_qhat=0.3),
    )
    return TestClient(api.app)


def test_valid_request_returns_decision(client: TestClient) -> None:
    r = client.post("/triage", json={"subject": "Refund", "body": "invoice billing payment"})
    assert r.status_code == 200
    data = r.json()
    assert data["queue"] in {"billing", "tech"}
    assert data["action"] in {"auto_route", "human_review"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["candidates"], list)


def test_invalid_body_is_422(client: TestClient) -> None:
    assert client.post("/triage", json={"subject": 5, "summarize": "nope"}).status_code == 422


def test_empty_text_is_422(client: TestClient) -> None:
    assert client.post("/triage", json={"subject": "  ", "body": ""}).status_code == 422


def test_api_matches_direct_library_call(client: TestClient) -> None:
    text = "Refund\ninvoice billing payment"
    response = client.post("/triage", json={"subject": "Refund", "body": "invoice billing payment"})

    model = api.state.queue_model
    assert model is not None
    policy = api.state.policy
    assert policy is not None
    from ticketintel.reliability import calibration
    from ticketintel.reliability.policy import decide

    proba = calibration.apply_temperature(
        model.predict_proba([text])[0].reshape(1, -1), policy.temperature
    )[0]
    expected = decide(proba, model.classes_, policy.abstention_threshold, policy.conformal_qhat)

    assert response.json()["queue"] == expected.queue
    assert response.json()["action"] == expected.action


def test_summarizer_outage_does_not_break_triage(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_: str) -> tuple[None, None, str]:
        return None, None, "ollama unreachable"

    monkeypatch.setattr(api, "_summarize", boom)
    r = client.post("/triage", json={"subject": "Refund", "body": "invoice", "summarize": True})
    assert r.status_code == 200
    assert r.json()["summary"] is None
    assert r.json()["summary_error"] == "ollama unreachable"
    assert r.json()["queue"]  # the routing decision still came back


def test_health_and_metrics(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model_loaded"] and health["policy_loaded"]

    client.post("/triage", json={"subject": "x", "body": "invoice billing"})
    body = client.get("/metrics").text
    assert "ticketintel_triage_decisions_total" in body
    assert "ticketintel_triage_latency_seconds" in body


def test_triage_without_models_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.state, "queue_model", None)
    monkeypatch.setattr(api.state, "policy", None)
    r = TestClient(api.app).post("/triage", json={"subject": "a", "body": "b"})
    assert r.status_code == 503
