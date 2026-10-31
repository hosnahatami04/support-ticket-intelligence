from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TicketClassifier(Protocol):
    """Shared interface every model tier implements.

    The reliability layer and the service depend only on this protocol, never on
    a concrete model type, so calibration and routing are model-agnostic.
    """

    @property
    def classes_(self) -> np.ndarray: ...

    def fit(self, texts: list[str], labels: list[str]) -> TicketClassifier: ...

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> TicketClassifier: ...
