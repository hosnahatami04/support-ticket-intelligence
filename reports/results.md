# Results

Every number here comes from a single pass over the test split with the model and the thresholds frozen beforehand. The thresholds were fitted on the calibration split only.

## Provenance

- dataset_id: `Tobi-Bueck/customer-support-tickets`
- dataset_revision: `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`
- processed_data_sha256: `d5a1b29844089e162cfdbe62a45d5e8f81819e6d2843249f0180147d9fc03a9f`
- git_revision: `c35531b2b2d8c0770c3a0bf610c1dba0dee1b3da`
- seed: `42`

## Queue routing (test)

| metric | value | target |
| --- | --- | --- |
| macro-F1 | 0.5812 [0.5424, 0.6172] | above baseline |
| lowest class recall | 0.3429 (Human Resources) | report |
| ECE after calibration | 0.0379 | <= 0.03 |
| auto-route error | 0.0333 | <= 0.05 |
| auto-route coverage | 0.0766 | >= 0.60 |
| AURC | 0.2046 | lower is better |
| conformal coverage | 0.8979 | 0.90 +/- 0.02 |
| mean conformal set size | 3.2463 | report |
| novel-queue AUROC | 0.5745 | >= 0.75 |
| error if coverage were raised to 0.6000 | 0.2404 | the cost of the coverage target |

## Priority (test)

| metric | value |
| --- | --- |
| macro-F1 | 0.6169 [0.5909, 0.6407] |
| ordinal MAE | 0.4359 [0.4065, 0.4678] |

## Summarization (fixed 300-ticket sample)

| metric | value | target |
| --- | --- | --- |
| schema validity | 1.0000 | >= 0.98 |
| unsupported entities | 0.0000 | <= 0.02 |
| ungrounded quotes | 0.0567 | report |
| NLI entailed | 0.5016 | >= 0.90 |
| compression | 0.5154 | report |

## Ablations

| ablation | result |
| --- | --- |
| random vs group-aware split | 0.6871 vs 0.6140 (inflation 0.0731) |
| word vs word+char n-grams | 0.5833 vs 0.5780 (gain -0.0054) |
| ECE without vs with scaling | 0.0643 vs 0.0379 |
| threshold only vs + conformal, coverage | 0.0772 vs 0.0766 |
| threshold only vs + conformal, error | 0.0331 vs 0.0333 |

## Figures

- `reports/figures/confusion_matrix.png`
- `reports/figures/reliability.png`
- `reports/figures/risk_coverage.png`
- `reports/figures/conformal_set_size.png`
- `reports/figures/novel_queue.png`

## What didn't work

- A fine-tuned DistilBERT did not beat the TF-IDF baseline within the CPU
  budget. Given several epochs on a shortened, subsampled corpus it reached
  macro-F1 0.245 on queue against 0.578 for TF-IDF, at 268 MB and roughly
  fifteen times the latency. The lighter model was selected
  (docs/decisions/0005).
- Sentence embeddings also underperformed the baseline on both tasks
  (queue 0.292, priority 0.437), so the middle tier earned nothing either.
- Character n-grams did not help. On the final comparison word-only scored
  marginally higher than word plus character n-grams, so the extra feature
  space bought nothing measurable on this data.
- The conformal set barely changes the routing decision. Threshold-only and
  threshold-plus-conformal give nearly identical coverage and error, because
  the abstention threshold is already strict enough that the singleton
  condition rarely binds. The guarantee is still worth reporting, but it is
  not what is holding the error rate down.
- The 5% auto-route error target and the 60% coverage target cannot hold at
  once on this dataset. The error target is met comfortably, but only a
  small share of tickets clear the bar. The cost of reaching the coverage
  target is measured and stated in the queue table above; the trade-off is
  reported rather than tuned away.
- Max calibrated probability barely separates the held-out queue from known
  ones, so novel-queue AUROC falls well short of 0.75. Confidence alone is a
  weak novelty signal here.
- Temperature scaling improved calibration but not enough to clear the 0.03
  ECE target on test.
- The first faithfulness measurement scored 0.22 because the summary fields
  are noun phrases and an entailment model scores propositions. Restating
  each field as a declarative claim raised it to 0.42. That was a fix to the
  measurement, not to the summaries.
