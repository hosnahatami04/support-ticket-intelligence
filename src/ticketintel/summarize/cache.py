from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from ticketintel.logging import get_logger

logger = get_logger(__name__)


class SummaryCache:
    """Content-addressed cache of model outputs.

    The key covers everything that can change an output - the ticket text, the
    prompt version, the exact model weights and the sampling parameters - so a
    cached summary is only reused when it would be regenerated identically.
    Evaluation reads from this cache, which is what makes the KPIs reproducible.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        ticket_text: str,
        prompt_version: str,
        model_tag: str,
        model_digest: str,
        params: Mapping[str, object],
        prompt_text: str | None = None,
    ) -> str:
        # The prompt's CONTENT is part of the key, not just its version label.
        # Editing a prompt file in place would otherwise keep serving the old
        # outputs from cache, so an evaluation would silently report results
        # the current prompt never produced.
        material = json.dumps(
            {
                "ticket": ticket_text,
                "prompt_version": prompt_version,
                "prompt_sha256": hashlib.sha256((prompt_text or "").encode()).hexdigest(),
                "model_tag": model_tag,
                "model_digest": model_digest,
                "params": params,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict[str, object] | None:
        path = self._path(key)
        if not path.exists():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None

    def put(self, key: str, value: dict[str, object]) -> None:
        self._path(key).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
