from __future__ import annotations

from pathlib import Path

import pytest

from ticketintel.summarize.cache import SummaryCache
from ticketintel.summarize.client import OllamaClient, OllamaConfig
from ticketintel.summarize.schema import TicketSummary
from ticketintel.summarize.summarizer import (
    SummarizationError,
    render_prompt,
    summarize,
    summarize_from_cache_only,
)

TEMPLATE = "Summarise:\n{ticket}"


class _CountingClient(OllamaClient):
    def __init__(self, responses: list[dict[str, object]]):
        super().__init__(OllamaConfig(model="m", temperature=0.0))
        self.responses = responses
        self.calls = 0

    def generate_json(self, prompt: str) -> dict[str, object]:
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def test_render_prompt_inserts_ticket() -> None:
    assert render_prompt(TEMPLATE, "disk full") == "Summarise:\ndisk full"


def test_repeat_request_is_served_from_cache(tmp_path: Path) -> None:
    client = _CountingClient([{"issue": "disk full"}])
    cache = SummaryCache(tmp_path)

    first = summarize("ticket text", client, TEMPLATE, cache, "digest")
    second = summarize("ticket text", client, TEMPLATE, cache, "digest")

    assert first == second
    assert client.calls == 1  # the second call never reached the server


def test_invalid_output_is_retried_once_then_succeeds(tmp_path: Path) -> None:
    client = _CountingClient([{"urgency_signals": "not a list"}, {"issue": "ok"}])
    summary = summarize("t", client, TEMPLATE, SummaryCache(tmp_path), "digest")
    assert summary.issue == "ok"
    assert client.calls == 2


def test_two_invalid_outputs_raise(tmp_path: Path) -> None:
    client = _CountingClient([{"urgency_signals": 1}, {"urgency_signals": 2}])
    with pytest.raises(SummarizationError, match="after 2 attempts"):
        summarize("t", client, TEMPLATE, SummaryCache(tmp_path), "digest")
    assert client.calls == 2


def test_cache_only_path_returns_none_on_miss(tmp_path: Path) -> None:
    cache = SummaryCache(tmp_path)
    assert summarize_from_cache_only("absent", cache, "m", "digest", 0.0) is None


def test_cache_only_path_reads_stored_summary(tmp_path: Path) -> None:
    cache = SummaryCache(tmp_path)
    client = _CountingClient([{"issue": "stored"}])
    summarize("t", client, TEMPLATE, cache, "digest")

    loaded = summarize_from_cache_only("t", cache, "m", "digest", 0.0)
    assert isinstance(loaded, TicketSummary)
    assert loaded.issue == "stored"
    assert client.calls == 1  # evaluation path did not call the model
