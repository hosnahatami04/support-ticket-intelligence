# 0007: Local summarization and reference-free faithfulness

## Context
The product must produce a structured summary of each ticket and must be able
to say how faithful that summary is to the source. No reference summaries
exist for this dataset, so any metric that compares against a gold summary
(ROUGE and friends) is meaningless here.

## Decision
- **A local model, called directly over HTTP.** Summaries come from
  `qwen2.5:3b-instruct` served by a local Ollama instance, at temperature 0.
  The client is a small hand-written HTTP wrapper with a timeout, a bounded
  retry and explicit errors; no orchestration framework (section 2.3).
- **Structured output with one retry.** The model is asked for JSON and the
  reply is validated against a Pydantic schema. One malformed reply is retried
  once; a second failure is recorded as a failure rather than silently dropped.
  Fields the ticket does not state must be `null`, which makes "the model did
  not know" distinguishable from "the model invented something".
- **A deterministic cache keyed by everything that can change the output** -
  ticket text, prompt version, model tag, model digest and sampling
  parameters. Evaluation reads only from this cache, so the reported KPIs are
  reproducible without re-running the model.
- **Reference-free faithfulness, in two layers.**
  1. A deterministic entity check: every number, email, URL and identifier-like
     code in the summary must appear in the ticket. Numbers are compared in a
     normalised form so `1,200` and `1200` match.
  2. Sentence-level NLI with the ticket as premise, using
     `cross-encoder/nli-deberta-v3-small`.
- **The NLI model's own accuracy is not asserted by the agent.** A balanced
  sample of 30 entailed and 30 not-entailed sentences is exported for the
  repository owner to review by hand; no claim about NLI accuracy is made
  until that review exists (section 4.2).

## Rejected options
- ROUGE or BERTScore against a reference: there is no reference to compare to.
- A cloud model API: ruled out by the project's constraints (section 2.3).
- Trusting the NLI label as ground truth: the model has its own error rate, and
  reporting it as if it were correct would overstate what was measured.

## Consequence
Summary KPIs (schema validity, unsupported-entity rate, NLI-entailed rate,
compression, latency) are computed from the cache and recorded in
`reports/interim/summarization_eval.json`. The human-review sample ships
alongside it and is the only basis on which NLI accuracy may later be claimed.

## Measured outcome and what the NLI number means

On the fixed 300-ticket test sample (`reports/interim/summarization_eval.json`):

| KPI | Result | Target |
| --- | --- | --- |
| Schema validity | 1.00 | >= 0.98 |
| Unsupported entities | 0.00 | <= 0.02 |
| NLI entailed | 0.42 | >= 0.90 |
| Compression | 0.49 | report |
| Latency p50 / p95 | 1.67 s / 2.54 s | report |

The NLI rate misses its target and is reported as measured. Two findings sit
behind it, and they are different in kind:

- **Hypothesis formatting was a measurement bug, now fixed.** The first run
  fed the raw fields to the NLI model and scored 0.22. Those fields are noun
  phrases, and an entailment model scores propositions: the same content is
  `neutral` as "Disruptions" and `entailment` as "The issue is disruptions."
  Scoring declarative claims instead raised the rate to 0.42. The fix changed
  the measurement, not the summaries.
- **The remaining gap mixes NLI error with real summary weakness.** Some
  flagged claims are supported almost verbatim by the ticket ("The affected
  area is healthcare provider's systems" against "impacting the healthcare
  provider's systems") and are model false negatives. Others are genuinely
  poor: the model sometimes fills `urgency_signals` with topic labels such as
  "project Management", which are not urgency signals at all.

Because the two cannot be separated automatically, no claim is made here about
the NLI model's accuracy or about the true faithfulness rate. The balanced
60-sentence sample in `reports/interim/nli_review_sample.json` is left for the
repository owner to adjudicate, with `human_verdict` deliberately blank.
