# 0005: Transformer architecture and final model selection

## Context
Phase 3 fine-tunes DistilBERT on CPU within a time budget and compares three
tiers (TF-IDF, sentence embeddings, DistilBERT) on validation, so later phases
can commit to one model for queue and one for priority.

## Decision
- **Two separate single-task models, not one shared encoder with two heads.**
  The `TicketClassifier` protocol is single-task (`predict_proba` returns one
  task's distribution), and the reliability layer calibrates and routes queue
  and priority independently. Two models keep the protocol and the service
  simple; the extra training cost is bounded by the shared time budget, split
  evenly across the two tasks.
- **Budgeted CPU training.** Each epoch is timed; the loop refuses to start an
  epoch it cannot finish within the per-task budget, and stops early on
  validation macro-F1 with patience 2. `max_length` is 256. If the budget were
  too small for even one epoch, the fallback is to lower `max_length` or
  subsample the training set, never to overrun.
- **Selection rule.** For each task, the transformer is chosen for later phases
  only if its validation macro-F1 beats the best lighter tier by at least 1.0
  point; otherwise the lighter tier is kept. The rule is applied per task from
  the measured comparison table (`reports/interim/model_comparison.md`).

## Rejected options
- A single two-head model: lower compute, but breaks the single-task protocol
  and couples the two tasks' calibration.
- Unbudgeted training to convergence: violates the CPU time budget (section 4.2).

## Measured outcome
Validation macro-F1 from the three-tier comparison
(`reports/interim/model_comparison.md`). The transformer tier was trained within
the CPU budget (`max_length=64`, a 1,200-row train subsample, six epochs of
~72s each) so it reaches several epochs for a fair comparison; the lighter tiers
train on the full train split. All tiers are evaluated on the full validation
split.

| task | tfidf | embeddings | transformer |
| --- | --- | --- | --- |
| queue | 0.578 | 0.292 | 0.245 |
| priority | 0.610 | 0.437 | 0.395 |

TF-IDF wins both tasks by a wide margin, is the smallest model (6.6 / 3.3 MB vs
268 MB), and has the lowest single-ticket latency (~2-3 ms p95 vs 17-44 ms).

**Selected model: TF-IDF for both queue and priority.** The transformer does not
come within 1.0 macro-F1 point of the lighter tier — it is far below it — so the
selection rule keeps TF-IDF. This is the expected consequence of the CPU budget:
DistilBERT cannot fine-tune enough on CPU, within a few hours, to match a tuned
TF-IDF on this dataset, even given several epochs. Reducing `max_length` and
subsampling (section 4.2) gave the transformer a fair multi-epoch run rather
than a single undertrained epoch, and the conclusion held.

## Consequence
The reliability and service layers load the TF-IDF models for both tasks. The
transformer code and comparison remain in the repository as the measured
baseline that justifies the choice.
