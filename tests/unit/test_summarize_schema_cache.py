from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ticketintel.summarize.cache import SummaryCache
from ticketintel.summarize.schema import TicketSummary


def test_nulls_are_allowed() -> None:
    s = TicketSummary.model_validate({"issue": None, "customer_request": None})
    assert s.issue is None
    assert s.urgency_signals == []


def test_wrong_field_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TicketSummary.model_validate({"urgency_signals": "should be a list"})


def test_text_fields_skips_empty_values() -> None:
    s = TicketSummary(issue="disk full", customer_request=None, urgency_signals=["asap"])
    assert s.text_fields() == ["disk full", "asap"]


def test_cache_round_trip_and_miss(tmp_path: Path) -> None:
    cache = SummaryCache(tmp_path)
    key = cache.key("ticket", "summary_v1", "m", "digest", {"temperature": 0.0})
    assert cache.get(key) is None
    cache.put(key, {"issue": "x"})
    assert cache.get(key) == {"issue": "x"}


def test_prompt_version_changes_the_key(tmp_path: Path) -> None:
    cache = SummaryCache(tmp_path)
    args = ("ticket", "m", "digest", {"temperature": 0.0})
    k1 = cache.key(args[0], "summary_v1", *args[1:])
    k2 = cache.key(args[0], "summary_v2", *args[1:])
    assert k1 != k2


def test_model_digest_changes_the_key(tmp_path: Path) -> None:
    cache = SummaryCache(tmp_path)
    k1 = cache.key("t", "summary_v1", "m", "digest-a", {"temperature": 0.0})
    k2 = cache.key("t", "summary_v1", "m", "digest-b", {"temperature": 0.0})
    assert k1 != k2


def test_claims_are_declarative_sentences() -> None:
    s = TicketSummary(
        issue="Disruptions",
        customer_request="A refund",
        affected_area="Billing portal",
        urgency_signals=["deadline is tomorrow"],
    )
    claims = s.claims()
    assert claims == [
        "The issue is disruptions.",
        "The customer requests a refund.",
        "The affected area is billing portal.",
        "The ticket indicates deadline is tomorrow.",
    ]


def test_claims_preserve_acronyms() -> None:
    s = TicketSummary(affected_area="VPN gateway")
    assert s.claims() == ["The affected area is VPN gateway."]


def test_claims_skip_missing_fields() -> None:
    assert TicketSummary().claims() == []
