from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.ml

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ticketintel.models.transformer import TrainConfig, TransformerClassifier  # noqa: E402

# A tiny, locally-built model keeps these tests fast and offline.
TINY_MODEL = "hf-internal-testing/tiny-random-distilbert"


def _dataset() -> tuple[list[str], list[str]]:
    texts = [
        "refund invoice billing payment charge",
        "billing refund payment invoice overcharge",
        "server crash timeout error exception logs",
        "timeout crash server error stack trace",
    ]
    labels = ["billing", "billing", "tech", "tech"]
    return texts, labels


def _config() -> TrainConfig:
    return TrainConfig(model_name=TINY_MODEL, max_length=32, batch_size=2, max_epochs=1)


def test_training_overfits_tiny_set() -> None:
    # A randomly-initialised tiny model starts near ln(2) and a handful of steps
    # can wobble either way, so assert the stronger, non-flaky property: given
    # enough steps the model drives training loss well below the starting point
    # on a clearly separable set.
    import torch.nn.functional as f

    torch.manual_seed(0)
    texts, labels = _dataset()
    model = TransformerClassifier(_config())
    model._labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(model._labels)}
    encoded = model._encode(texts)
    y = torch.tensor([label_to_id[label] for label in labels])

    net = model._init_model(len(model._labels))
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=5e-3)

    def loss_now() -> float:
        out = net(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        return float(f.cross_entropy(out.logits, y))

    before = loss_now()
    for _ in range(30):
        opt.zero_grad()
        out = net(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], labels=y
        )
        out.loss.backward()
        opt.step()
    assert loss_now() < before * 0.5


def test_predict_proba_shape_and_sum() -> None:
    texts, labels = _dataset()
    model = TransformerClassifier(_config()).fit(texts, labels)
    proba = model.predict_proba(texts)
    assert proba.shape == (len(texts), len(set(labels)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    texts, labels = _dataset()
    model = TransformerClassifier(_config()).fit(texts, labels)
    before = model.predict_proba(texts)
    path = tmp_path / "ckpt"
    model.save(path)
    reloaded = TransformerClassifier.load(path)
    assert np.allclose(reloaded.predict_proba(texts), before, atol=1e-5)
    assert list(reloaded.classes_) == list(model.classes_)


def test_long_text_is_truncated() -> None:
    model = TransformerClassifier(_config())
    model._labels = ["a"]
    long_text = "word " * 5000
    encoded = model._encode([long_text])
    assert encoded["input_ids"].shape[1] <= _config().max_length


def test_fit_without_validation_runs_all_epochs(caplog: pytest.LogCaptureFixture) -> None:
    # Regression: without a validation set the loop must run every epoch and keep
    # the final weights, not stop after `early_stopping_patience` epochs.
    import logging

    texts, labels = _dataset()
    config = TrainConfig(
        model_name=TINY_MODEL,
        max_length=32,
        batch_size=2,
        max_epochs=3,
        early_stopping_patience=1,
    )
    with caplog.at_level(logging.INFO, logger="ticketintel.models.transformer"):
        model = TransformerClassifier(config).fit(texts, labels)

    epochs_done = sum(1 for r in caplog.records if r.getMessage() == "epoch done")
    assert epochs_done == 3  # all max_epochs ran despite patience=1 and no val
    assert model.predict_proba(texts).shape == (len(texts), len(set(labels)))
