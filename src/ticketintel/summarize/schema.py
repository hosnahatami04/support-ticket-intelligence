from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TicketSummary(BaseModel):
    """Structured summary of one support ticket (section 3.3, task C).

    Every field is optional at the value level: the prompt instructs the model
    to emit null when the ticket does not state something, which is safer than
    inventing a value and is what the faithfulness check expects.
    """

    issue: str | None = Field(default=None, description="What is wrong, in one sentence.")
    customer_request: str | None = Field(default=None, description="What the customer asks for.")
    affected_area: str | None = Field(default=None, description="Product, system or component.")
    urgency_signals: list[str] = Field(
        default_factory=list, description="Phrases in the ticket that signal urgency."
    )

    @field_validator("issue", "customer_request", "affected_area", mode="before")
    @classmethod
    def _join_list(cls, value: object) -> object:
        """Accept a list where one string was asked for.

        A ticket that names two products ("Zoho CRM and Microsoft Dynamics
        365") genuinely has two affected areas, and the model answers with a
        list. That is the right information in the wrong container: rejecting
        it discarded a correct summary, and because decoding is deterministic
        at temperature 0 the retry produced the identical list, so the ticket
        could never succeed. Joining is safe for faithfulness - every element
        still has to survive the entity check.
        """
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(parts) if parts else None
        return value

    def text_fields(self) -> list[str]:
        # The free-text the deterministic entity check runs over.
        parts = [self.issue, self.customer_request, self.affected_area, *self.urgency_signals]
        return [p for p in parts if p]

    def claims(self) -> list[str]:
        # Declarative restatements of each field, for NLI. An entailment model
        # scores propositions, not noun phrases: the same content reads as
        # "neutral" as a bare fragment ("Disruptions") and as "entailment" as a
        # sentence ("The issue is disruptions."), so the fields are turned into
        # sentences before they are scored.
        out: list[str] = []
        if self.issue:
            out.append(f"The issue is {_lower_first(self.issue)}.")
        if self.customer_request:
            out.append(f"The customer requests {_lower_first(self.customer_request)}.")
        if self.affected_area:
            out.append(f"The affected area is {_lower_first(self.affected_area)}.")
        for signal in self.urgency_signals:
            if signal:
                out.append(f"The ticket indicates {_lower_first(signal)}.")
        return out


def _lower_first(text: str) -> str:
    # Lower-case only an ordinary capitalised word, so acronyms and product
    # names keep their casing when embedded mid-sentence.
    stripped = text.strip().rstrip(".")
    # Two leading capital LETTERS means an acronym ("VPN"); "A refund" is not
    # one, so the check must not be satisfied by a capital followed by a space.
    if len(stripped) >= 2 and stripped[0].isupper() and stripped[1].isupper():
        return stripped
    return stripped[:1].lower() + stripped[1:]
