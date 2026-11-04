from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.logging import get_logger
from ticketintel.summarize import faithfulness
from ticketintel.summarize.cache import SummaryCache
from ticketintel.summarize.client import OllamaClient, OllamaConfig
from ticketintel.summarize.nli import CrossEncoderScorer, EntailmentScorer, entailed_fraction
from ticketintel.summarize.schema import TicketSummary
from ticketintel.summarize.summarizer import SummarizationError, load_prompt, summarize

logger = get_logger(__name__)


@dataclass
class SummarizerSettings:
    model: str
    host: str
    temperature: float
    timeout_s: float
    max_retries: int
    prompt_version: str
    eval_sample_size: int
    eval_seed: int

    @classmethod
    def load(cls, settings: Settings) -> SummarizerSettings:
        raw = yaml.safe_load(
            (settings.paths.configs / "summarizer.yaml").read_text(encoding="utf-8")
        )
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def sample_tickets(settings: Settings, cfg: SummarizerSettings) -> pd.DataFrame:
    frame = pd.read_parquet(settings.paths.data_processed / "tickets.parquet")
    test = frame.loc[frame[schema.SPLIT] == "test"]
    n = min(cfg.eval_sample_size, len(test))
    return test.sample(n=n, random_state=cfg.eval_seed).reset_index(drop=True)


def generate_summaries(settings: Settings, cfg: SummarizerSettings) -> dict[str, int]:
    # Populates the cache. Evaluation itself reads only from the cache, so this
    # step is the only one that needs the model server.
    client = OllamaClient(
        OllamaConfig(
            model=cfg.model,
            host=cfg.host,
            temperature=cfg.temperature,
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )
    )
    digest = client.model_digest()
    template = load_prompt(Path("prompts"), cfg.prompt_version)
    cache = SummaryCache(settings.paths.artifacts / "summary_cache")

    tickets = sample_tickets(settings, cfg)
    ok = 0
    failures: list[dict[str, str]] = []
    for text in tickets[schema.TEXT]:
        try:
            summarize(text, client, template, cache, digest, cfg.prompt_version)
            ok += 1
        except SummarizationError as exc:
            # Keep why it failed. Counting failures without their reason is what
            # makes a persistent one impossible to diagnose from the report.
            failures.append({"reason": str(exc), "ticket_excerpt": text[:200]})
            logger.warning(
                "summary failed", extra={"reason": str(exc), "ticket_excerpt": text[:200]}
            )

    if failures:
        write_failures(failures, settings)
    logger.info("summaries produced", extra={"ok": ok, "failed": len(failures)})
    return {"ok": ok, "failed": len(failures)}


def write_failures(failures: list[dict[str, str]], settings: Settings) -> Path:
    out_dir = settings.paths.reports / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summarization_failures.json"
    path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return path


def evaluate(
    settings: Settings, cfg: SummarizerSettings, scorer: EntailmentScorer | None = None
) -> dict[str, object]:
    from ticketintel.summarize.summarizer import summarize_from_cache_only

    cache = SummaryCache(settings.paths.artifacts / "summary_cache")
    client = OllamaClient(OllamaConfig(model=cfg.model, host=cfg.host))
    digest = client.model_digest()
    template = load_prompt(Path("prompts"), cfg.prompt_version)
    tickets = sample_tickets(settings, cfg)
    scorer = scorer or CrossEncoderScorer()

    valid = 0
    unsupported_entity_count = 0
    ungrounded_quote_count = 0
    compressions: list[float] = []
    entail_fractions: list[float] = []
    review_rows: list[dict[str, object]] = []

    for text in tickets[schema.TEXT]:
        summary = summarize_from_cache_only(
            text, cache, cfg.model, digest, cfg.temperature, cfg.prompt_version, template
        )
        if summary is None:
            continue
        valid += 1

        fields = summary.text_fields()
        if not faithfulness.check_entities(fields, text).is_faithful:
            unsupported_entity_count += 1
        # urgency_signals are meant to be the ticket's own wording.
        if faithfulness.ungrounded_quotes(summary.urgency_signals, text):
            ungrounded_quote_count += 1

        summary_text = " ".join(fields)
        compressions.append(len(summary_text.split()) / max(len(text.split()), 1))

        # NLI scores declarative claims, not the raw field fragments.
        sentences = [s for c in summary.claims() for s in faithfulness.split_sentences(c)]
        fraction, flags = entailed_fraction(scorer, text, sentences)
        entail_fractions.append(fraction)
        for sentence, entailed in zip(sentences, flags, strict=True):
            review_rows.append(
                {
                    "sentence": sentence,
                    "nli_label": "entailed" if entailed else "not_entailed",
                    "ticket_excerpt": text[:300],
                }
            )

    total = len(tickets)
    return {
        "sample_size": total,
        "summaries_in_cache": valid,
        "schema_valid_rate": valid / total if total else 0.0,
        "unsupported_entity_rate": unsupported_entity_count / valid if valid else 0.0,
        "ungrounded_quote_rate": ungrounded_quote_count / valid if valid else 0.0,
        "nli_entailed_rate": float(np.mean(entail_fractions)) if entail_fractions else 0.0,
        "compression_ratio": float(np.mean(compressions)) if compressions else 0.0,
        "_review_rows": review_rows,
    }


def measure_latency(settings: Settings, cfg: SummarizerSettings, n: int = 40) -> dict[str, float]:
    # Timed against the live model, so this is excluded from the cached path.
    # n=40 rather than 10: a p95 taken from ten samples is essentially the
    # maximum of ten draws, and a single unrelated stall on the host then lands
    # in the report as a measured latency. Observed swings of 3.1 s to 6.3 s on
    # an unchanged code path came from exactly that.
    client = OllamaClient(
        OllamaConfig(
            model=cfg.model, host=cfg.host, temperature=cfg.temperature, timeout_s=cfg.timeout_s
        )
    )
    template = load_prompt(Path("prompts"), cfg.prompt_version)
    tickets = sample_tickets(settings, cfg)[schema.TEXT].tolist()[:n]
    timings: list[float] = []
    for text in tickets:
        start = time.perf_counter()
        try:
            client.generate_json(template.replace("{ticket}", text))
        except Exception:  # a failed call is still a latency sample we exclude
            continue
        timings.append((time.perf_counter() - start) * 1000)
    if not timings:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "samples": 0}
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "max_ms": float(np.max(timings)),
        "samples": float(len(timings)),
    }


def write_review_file(
    rows: list[dict[str, object]], settings: Settings, per_class: int = 30
) -> Path:
    # A balanced sample for the user's manual NLI review (section 4.2). The
    # agent does not judge these itself.
    rng = np.random.default_rng(0)
    out_dir = settings.paths.reports / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "nli_review_sample.json"

    picked: list[dict[str, object]] = []
    for label in ("entailed", "not_entailed"):
        pool = [r for r in rows if r["nli_label"] == label]
        if pool:
            idx = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
            picked.extend(pool[int(i)] | {"human_verdict": ""} for i in idx)
    path.write_text(json.dumps(picked, indent=2), encoding="utf-8")
    return path


def write_report(results: dict[str, object], settings: Settings) -> Path:
    out_dir = settings.paths.reports / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summarization_eval.json"
    public = {k: v for k, v in results.items() if not k.startswith("_")}
    path.write_text(json.dumps(public, indent=2, sort_keys=True), encoding="utf-8")
    return path


def summary_to_dict(summary: TicketSummary) -> dict[str, object]:
    return summary.model_dump()
