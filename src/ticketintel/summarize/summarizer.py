from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ticketintel.logging import get_logger
from ticketintel.summarize.cache import SummaryCache
from ticketintel.summarize.client import OllamaClient, OllamaError
from ticketintel.summarize.schema import TicketSummary

logger = get_logger(__name__)

PROMPT_VERSION = "summary_v1"


class SummarizationError(RuntimeError):
    pass


def load_prompt(prompts_dir: Path, version: str = PROMPT_VERSION) -> str:
    return (prompts_dir / f"{version}.txt").read_text(encoding="utf-8")


def render_prompt(template: str, ticket_text: str) -> str:
    return template.replace("{ticket}", ticket_text)


def summarize(
    ticket_text: str,
    client: OllamaClient,
    template: str,
    cache: SummaryCache,
    model_digest: str,
    prompt_version: str = PROMPT_VERSION,
) -> TicketSummary:
    params = {"temperature": client.config.temperature}
    key = cache.key(ticket_text, prompt_version, client.config.model, model_digest, params)

    cached = cache.get(key)
    if cached is not None:
        return TicketSummary.model_validate(cached)

    prompt = render_prompt(template, ticket_text)
    last_error: Exception | None = None
    # One retry: a model occasionally emits a field of the wrong shape, and a
    # single regeneration is cheap. A second failure is recorded, not hidden.
    for attempt in range(2):
        try:
            raw = client.generate_json(prompt)
            summary = TicketSummary.model_validate(raw)
        except (OllamaError, ValidationError) as exc:
            last_error = exc
            logger.info("summary attempt failed", extra={"attempt": attempt, "error": str(exc)})
            continue
        cache.put(key, summary.model_dump())
        return summary

    raise SummarizationError(f"summary failed after 2 attempts: {last_error}")


def summarize_from_cache_only(
    ticket_text: str,
    cache: SummaryCache,
    model_tag: str,
    model_digest: str,
    temperature: float,
    prompt_version: str = PROMPT_VERSION,
) -> TicketSummary | None:
    # Evaluation path: never calls the model, so KPI numbers are reproducible
    # from the cache alone (section 2.4).
    key = cache.key(
        ticket_text, prompt_version, model_tag, model_digest, {"temperature": temperature}
    )
    cached = cache.get(key)
    return TicketSummary.model_validate(cached) if cached is not None else None
