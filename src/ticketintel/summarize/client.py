from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from ticketintel.logging import get_logger

logger = get_logger(__name__)

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Raised when the local model server cannot be reached or returns an error."""


@dataclass(frozen=True)
class OllamaConfig:
    model: str
    host: str = DEFAULT_HOST
    temperature: float = 0.0
    timeout_s: float = 120.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0


class OllamaClient:
    """Minimal HTTP client for a local Ollama server.

    Written directly against the HTTP API rather than through an orchestration
    framework (section 2.3), so the only failure modes are the ones handled here.
    """

    def __init__(self, config: OllamaConfig):
        self.config = config

    def generate_json(self, prompt: str) -> dict[str, object]:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.config.temperature},
        }
        raw = self._post("/api/generate", payload)
        response = raw.get("response")
        if not isinstance(response, str):
            raise OllamaError("model response did not contain a 'response' string")
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"model did not return valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("model returned JSON that is not an object")
        return parsed

    def model_digest(self) -> str:
        # The digest pins the exact weights, so cached summaries are invalidated
        # when the model changes underneath us.
        models = self._post("/api/tags", None, method="GET").get("models")
        if not isinstance(models, list):
            raise OllamaError("/api/tags did not return a model list")
        for entry in models:
            if isinstance(entry, dict) and entry.get("model") == self.config.model:
                digest = entry.get("digest")
                if isinstance(digest, str):
                    return digest
        raise OllamaError(f"model {self.config.model!r} is not available on the server")

    def _post(
        self, path: str, payload: dict[str, object] | None, method: str = "POST"
    ) -> dict[str, object]:
        url = f"{self.config.host}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(
                url, data=data, method=method, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                    body = json.loads(response.read().decode())
                if not isinstance(body, dict):
                    raise OllamaError(f"{path} returned JSON that is not an object")
                return body
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.info(
                    "ollama request failed",
                    extra={"path": path, "attempt": attempt, "error": str(exc)},
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_s * (attempt + 1))

        raise OllamaError(
            f"{path} failed after {self.config.max_retries + 1} attempts: {last_error}"
        )
