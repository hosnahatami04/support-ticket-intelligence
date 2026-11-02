from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from ticketintel.config import Settings, load_settings
from ticketintel.logging import configure_logging, get_logger
from ticketintel.reliability import calibration
from ticketintel.reliability.policy import decide
from ticketintel.service.schemas import (
    FaithfulnessReport,
    HealthResponse,
    TriageRequest,
    TriageResponse,
)

if TYPE_CHECKING:
    from ticketintel.models.tfidf import TfidfClassifier

logger = get_logger(__name__)

TRIAGE_DECISIONS = Counter(
    "ticketintel_triage_decisions_total", "Triage decisions by action", ["action"]
)
TRIAGE_LATENCY = Histogram(
    "ticketintel_triage_latency_seconds", "Triage request latency in seconds"
)


@dataclass
class Policy:
    temperature: float
    abstention_threshold: float
    conformal_qhat: float

    @classmethod
    def load(cls, settings: Settings) -> Policy:
        raw = yaml.safe_load((settings.paths.configs / "policy.yaml").read_text(encoding="utf-8"))
        queue = raw["queue"]
        return cls(
            temperature=float(queue["temperature"]),
            abstention_threshold=float(queue["abstention_threshold"]),
            conformal_qhat=float(queue["conformal_qhat"]),
        )


@dataclass
class ServiceState:
    settings: Settings
    queue_model: TfidfClassifier | None = None
    priority_model: TfidfClassifier | None = None
    policy: Policy | None = None

    @property
    def ready(self) -> bool:
        return self.queue_model is not None and self.policy is not None


state = ServiceState(settings=load_settings())


def _load_models(st: ServiceState) -> None:
    # Loaded once at startup so a request never pays model-loading cost.
    from ticketintel.models.tfidf import TfidfClassifier

    artifacts = st.settings.paths.artifacts
    for name, attr in (
        ("tfidf_queue.joblib", "queue_model"),
        ("tfidf_priority.joblib", "priority_model"),
    ):
        path = artifacts / name
        if path.exists():
            setattr(st, attr, TfidfClassifier.load(path))
        else:
            logger.info("model artifact missing", extra={"path": str(path)})
    policy_path = st.settings.paths.configs / "policy.yaml"
    if policy_path.exists():
        st.policy = Policy.load(st.settings)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    _load_models(state)
    yield


app = FastAPI(title="ticketintel triage", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.ready else "degraded",
        model_loaded=state.queue_model is not None,
        policy_loaded=state.policy is not None,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _predict_label(model: Any, text: str) -> tuple[str, np.ndarray]:
    proba = model.predict_proba([text])[0]
    return str(model.classes_[int(proba.argmax())]), proba


def _summarize(text: str) -> tuple[Any, FaithfulnessReport | None, str | None]:
    # The summarizer is optional: if the local model server is unreachable the
    # triage decision still returns, with the failure reported explicitly.
    from ticketintel.summarize import faithfulness as fcheck
    from ticketintel.summarize.cache import SummaryCache
    from ticketintel.summarize.client import OllamaClient, OllamaConfig, OllamaError
    from ticketintel.summarize.evaluate import SummarizerSettings
    from ticketintel.summarize.summarizer import SummarizationError, load_prompt, summarize

    try:
        cfg = SummarizerSettings.load(state.settings)
        client = OllamaClient(
            OllamaConfig(
                model=cfg.model,
                host=cfg.host,
                temperature=cfg.temperature,
                timeout_s=cfg.timeout_s,
                max_retries=cfg.max_retries,
            )
        )
        summary = summarize(
            text,
            client,
            load_prompt(Path("prompts"), cfg.prompt_version),
            SummaryCache(state.settings.paths.artifacts / "summary_cache"),
            client.model_digest(),
            cfg.prompt_version,
        )
    except (OllamaError, SummarizationError, OSError) as exc:
        logger.info("summarization unavailable", extra={"error": str(exc)})
        return None, None, str(exc)

    check = fcheck.check_entities(summary.text_fields(), text)
    quotes = fcheck.ungrounded_quotes(summary.urgency_signals, text)
    return (
        summary,
        FaithfulnessReport(
            unsupported_entities=check.unsupported,
            ungrounded_quotes=quotes,
            is_faithful=check.is_faithful and not quotes,
        ),
        None,
    )


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest) -> TriageResponse:
    start = time.perf_counter()
    text = request.text()
    if not text:
        raise HTTPException(status_code=422, detail="subject and body are both empty")
    if state.queue_model is None or state.policy is None:
        raise HTTPException(status_code=503, detail="model or policy is not loaded")

    queue_proba = state.queue_model.predict_proba([text])[0]
    calibrated = calibration.apply_temperature(
        queue_proba.reshape(1, -1), state.policy.temperature
    )[0]
    decision = decide(
        calibrated,
        state.queue_model.classes_,
        state.policy.abstention_threshold,
        state.policy.conformal_qhat,
    )

    priority = "unknown"
    if state.priority_model is not None:
        priority, _ = _predict_label(state.priority_model, text)

    summary, faith, summary_error = (None, None, None)
    if request.summarize:
        summary, faith, summary_error = _summarize(text)

    TRIAGE_DECISIONS.labels(action=decision.action).inc()
    TRIAGE_LATENCY.observe(time.perf_counter() - start)

    return TriageResponse(
        queue=decision.queue,
        priority=priority,
        confidence=decision.confidence,
        action=decision.action,
        candidates=decision.candidates,
        reason=decision.reason,
        summary=summary,
        faithfulness=faith,
        summary_error=summary_error,
    )
