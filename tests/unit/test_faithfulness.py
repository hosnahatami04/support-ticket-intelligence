from __future__ import annotations

from ticketintel.summarize.faithfulness import (
    check_entities,
    split_sentences,
    ungrounded_quotes,
)
from ticketintel.summarize.nli import entailed_fraction

SOURCE = (
    "Order 48219 has not arrived. I paid 1,200 euros on the invoice and wrote to "
    "billing@example.com twice. See https://status.example.com for the outage."
)


def test_faithful_summary_is_not_flagged() -> None:
    summary = ["Order 48219 is missing", "Customer paid 1200 euros", "billing@example.com"]
    assert check_entities(summary, SOURCE).is_faithful


def test_fabricated_number_is_flagged() -> None:
    result = check_entities(["Refund of 9999 euros requested"], SOURCE)
    assert not result.is_faithful
    assert "9999" in result.unsupported


def test_fabricated_email_is_flagged() -> None:
    result = check_entities(["Contact support@other.test"], SOURCE)
    assert "support@other.test" in result.unsupported


def test_fabricated_order_code_is_flagged() -> None:
    result = check_entities(["Order AB12345 is delayed"], SOURCE)
    assert "AB12345" in result.unsupported


def test_number_formatting_differences_are_tolerated() -> None:
    # "1,200" in the source and "1200" in the summary are the same quantity.
    assert check_entities(["paid 1200 euros"], SOURCE).is_faithful


def test_empty_summary_is_trivially_faithful() -> None:
    assert check_entities([], SOURCE).is_faithful


def test_split_sentences_handles_punctuation() -> None:
    assert split_sentences("A failed. B worked! C?") == ["A failed.", "B worked!", "C?"]


class _StubScorer:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[str]:
        self.seen = pairs
        return self.labels[: len(pairs)]


def test_entailed_fraction_counts_only_entailment() -> None:
    scorer = _StubScorer(["entailment", "neutral", "contradiction"])
    fraction, flags = entailed_fraction(scorer, SOURCE, ["a", "b", "c"])
    assert fraction == 1 / 3
    assert flags == [True, False, False]
    # Every hypothesis is checked against the full ticket as premise.
    assert all(premise == SOURCE for premise, _ in scorer.seen)


def test_entailed_fraction_with_no_sentences() -> None:
    fraction, flags = entailed_fraction(_StubScorer([]), SOURCE, [])
    assert fraction == 1.0
    assert flags == []


def test_ungrounded_quote_is_flagged() -> None:
    # The entity check cannot see this: no number, address or code is invented,
    # yet the phrase appears nowhere in the ticket.
    from ticketintel.summarize.faithfulness import ungrounded_quotes

    flagged = ungrounded_quotes(["system has been down since Monday"], SOURCE)
    assert flagged == ["system has been down since Monday"]


def test_quote_taken_from_the_ticket_is_not_flagged() -> None:
    from ticketintel.summarize.faithfulness import ungrounded_quotes

    assert ungrounded_quotes(["Order 48219 has not arrived"], SOURCE) == []


def test_light_paraphrase_is_tolerated() -> None:
    # Word overlap, not exact matching, so a reworded quote still passes.
    from ticketintel.summarize.faithfulness import ungrounded_quotes

    assert ungrounded_quotes(["paid 1,200 euros on the invoice"], SOURCE) == []


def test_empty_quotes_are_ignored() -> None:
    from ticketintel.summarize.faithfulness import ungrounded_quotes

    assert ungrounded_quotes(["", "   "], SOURCE) == []


_SRC = (
    "The system is not down. We were told the outage would last three days, "
    "but it has not started yet and nothing is blocked."
)
_PUNCT = "Service is down. Users are blocked, and billing failed!"


def test_a_quote_is_not_flagged_for_the_punctuation_beside_it() -> None:
    # Regression: tokens were split on whitespace only, so a source reading
    # "down." never matched a quote of "down" and every such verbatim quote was
    # counted as a fabrication. That inflated the reported rate.
    assert ungrounded_quotes(["down"], _PUNCT) == []
    assert ungrounded_quotes(["blocked"], _PUNCT) == []
    assert ungrounded_quotes(["billing failed"], _PUNCT) == []


def test_dropping_a_negation_is_flagged() -> None:
    # Regression: word overlap ignores order, so "the system is down" scored a
    # perfect match against "The system is not down" - the one case where a
    # quote asserts the opposite of the ticket.
    assert ungrounded_quotes(["the system is down"], _SRC) == ["the system is down"]


def test_adding_a_negation_is_flagged() -> None:
    assert ungrounded_quotes(["billing is not failed"], _PUNCT)


def test_a_preserved_negation_is_not_flagged() -> None:
    assert ungrounded_quotes(["the system is not down"], _SRC) == []
    assert ungrounded_quotes(["nothing is blocked"], _SRC) == []


def test_a_negation_in_an_unrelated_clause_does_not_flag_a_good_quote() -> None:
    # The source negates twice, but neither negation belongs to this phrase.
    # Comparing against the best-matching span rather than the span between the
    # first and last matching word is what keeps this from being flagged.
    assert ungrounded_quotes(["the outage would last three days"], _SRC) == []


def test_the_tickets_own_words_reassembled_into_a_new_phrase_is_flagged() -> None:
    # Every word is present, so overlap alone cannot tell this from a quote.
    # Adjacent word pairs can: a real quotation keeps its pairs intact.
    assert ungrounded_quotes(["down three days blocked outage"], _SRC)
    assert ungrounded_quotes(["failed billing users"], _PUNCT)


def test_a_verbatim_multiword_quote_survives_the_pair_check() -> None:
    assert ungrounded_quotes(["users are blocked"], _PUNCT) == []
