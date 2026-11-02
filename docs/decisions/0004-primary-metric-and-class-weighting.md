# 0004: Primary metric and class weighting

## Context
The queue classes are imbalanced (Technical Support has ~20x the volume of
General Inquiry). A metric and a training choice had to be fixed so every model
tier is compared on the same, defensible basis.

## Decision
- **Macro-F1 is the primary queue metric.** It averages F1 across queues with
  equal weight, so a model cannot look good by serving only the large queues.
  Accuracy and micro-averaged scores are reported nowhere as the headline.
- **Minimum per-class recall is reported alongside it**, so a rare queue that is
  silently ignored is visible even when macro-F1 looks acceptable.
- **Linear models train with `class_weight="balanced"`**, which reweights the
  loss by inverse class frequency. This is the simplest lever that lifts rare
  classes without resampling or synthetic data.
- **Priority is scored with both macro-F1 and ordinal MAE** over
  low=0, medium=1, high=2, because a high-vs-low error is worse than an
  adjacent-level error.

## Rejected options
- Accuracy as the headline: dominated by the largest queues, hides rare-class
  failure.
- Oversampling / SMOTE on rare queues: adds synthetic rows to an already
  synthetic dataset and complicates the pipeline for little expected gain over
  class weighting.

## Consequence
All model comparisons in this and later phases lead with macro-F1 and report
minimum per-class recall; priority additionally reports ordinal MAE.
