from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ticketintel.evaluation.metrics import macro_f1
from ticketintel.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL
    max_length: int = MAX_LENGTH
    batch_size: int = 16
    learning_rate: float = 5e-5
    max_epochs: int = 6
    # Stop before this wall-clock budget; the first epoch is timed and the loop
    # refuses to start an epoch it cannot finish within the budget (section 4.2).
    time_budget_s: float = 3 * 60 * 60
    early_stopping_patience: int = 2
    seed: int = 42


def _set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class _TextDataset:
    def __init__(self, encodings: dict[str, Any], labels: Any):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class TransformerClassifier:
    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._labels: list[str] = []

    @property
    def classes_(self) -> np.ndarray:
        if not self._labels:
            raise RuntimeError("model is not fitted")
        return np.asarray(self._labels)

    def _tokenizer_lazy(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self._tokenizer

    def _encode(self, texts: list[str]) -> dict[str, Any]:
        return dict(
            self._tokenizer_lazy()(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
        )

    def _init_model(self, num_labels: int) -> Any:
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=num_labels
        )

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        val_texts: list[str] | None = None,
        val_labels: list[str] | None = None,
    ) -> TransformerClassifier:
        import torch
        from torch.utils.data import DataLoader

        _set_seed(self.config.seed)
        self._labels = sorted(set(labels))
        label_to_id = {label: i for i, label in enumerate(self._labels)}

        train_ds: Any = _TextDataset(
            self._encode(texts),
            torch.tensor([label_to_id[label] for label in labels]),
        )
        loader: Any = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        model = self._init_model(len(self._labels))
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        best_f1 = -1.0
        best_state = None
        epochs_without_gain = 0
        start = time.monotonic()
        epoch_seconds = 0.0

        for epoch in range(self.config.max_epochs):
            elapsed = time.monotonic() - start
            # Refuse to begin an epoch we cannot finish inside the budget.
            if epoch_seconds and elapsed + epoch_seconds > self.config.time_budget_s:
                logger.info("stopping on time budget", extra={"epoch": epoch, "elapsed": elapsed})
                break

            epoch_start = time.monotonic()
            for batch in loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                out.loss.backward()
                optimizer.step()
            epoch_seconds = time.monotonic() - epoch_start
            self._model = model

            # Without a validation set there is no signal to early-stop or select
            # on, so train to max_epochs (or the budget) and keep the final
            # weights. Early stopping and best-checkpointing apply only with val.
            if val_texts is None or val_labels is None:
                logger.info("epoch done", extra={"epoch": epoch, "epoch_seconds": epoch_seconds})
                continue

            pred = self.classes_[self.predict_proba(val_texts).argmax(axis=1)]
            metric = macro_f1(val_labels, pred)
            model.train()
            logger.info(
                "epoch done",
                extra={"epoch": epoch, "val_macro_f1": metric, "epoch_seconds": epoch_seconds},
            )

            if metric > best_f1:
                best_f1 = metric
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= self.config.early_stopping_patience:
                    logger.info("early stopping", extra={"epoch": epoch})
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("model is not fitted")
        self._model.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.config.batch_size):
                batch = self._encode(texts[i : i + self.config.batch_size])
                logits = self._model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                probs.append(torch.softmax(logits, dim=-1).numpy())
        return np.vstack(probs)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer_lazy().save_pretrained(path)
        (path / "labels.json").write_text(json.dumps(self._labels), encoding="utf-8")
        (path / "train_config.json").write_text(json.dumps(vars(self.config)), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TransformerClassifier:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        config = TrainConfig(**json.loads((path / "train_config.json").read_text(encoding="utf-8")))
        model = cls(config)
        model._model = AutoModelForSequenceClassification.from_pretrained(path)
        model._tokenizer = AutoTokenizer.from_pretrained(path)
        model._labels = json.loads((path / "labels.json").read_text(encoding="utf-8"))
        return model
