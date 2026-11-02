from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ticketintel.summarize.schema import TicketSummary


class TriageRequest(BaseModel):
    subject: str = Field(default="")
    body: str = Field(default="")
    summarize: bool = False

    @field_validator("subject", "body")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    def text(self) -> str:
        # Same input construction as training (section 3.2), so the service and
        # the offline pipeline see identical text.
        return f"{self.subject}\n{self.body}".strip()


class FaithfulnessReport(BaseModel):
    unsupported_entities: list[str] = Field(default_factory=list)
    is_faithful: bool = True


class TriageResponse(BaseModel):
    queue: str
    priority: str
    confidence: float
    action: str
    candidates: list[str]
    reason: str
    summary: TicketSummary | None = None
    faithfulness: FaithfulnessReport | None = None
    summary_error: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    policy_loaded: bool
