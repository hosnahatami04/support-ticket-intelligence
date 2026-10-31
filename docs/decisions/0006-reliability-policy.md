# 0006: Calibration, abstention and the decision policy

## Context
The reliability layer decides when a ticket is routed automatically and when it
goes to a human. It has to turn a classifier's probabilities into a calibrated
confidence, an abstention rule, and a set-valued guarantee, all fit without ever
touching the test split.

## Decision
- **Temperature scaling, probability-space.** The scale is fit by a 1-D search
  minimising NLL on the calibration split. It operates on the probabilities
  `predict_proba` returns (converted to log space), not a model's internal
  logits, so the layer stays model-agnostic behind the classifier protocol.
- **ECE with 15 equal-width bins**, reported before and after scaling.
- **Selective prediction.** The confidence score is the calibrated max
  probability. The abstention threshold tau is the lowest confidence whose
  accepted set keeps auto-route error at or below 5%, chosen to maximise
  coverage. AURC summarises the whole risk-coverage trade-off.
- **Split conformal (LAC), alpha = 0.10.** Nonconformity is `1 - p(true)`, the
  threshold is the finite-sample-corrected quantile
  `ceil((n+1)(1-alpha))/n`, giving marginal coverage >= 90%.
- **Decision policy.** A ticket is auto-routed only when it is both confident
  (calibrated max prob >= tau) and unambiguous (its conformal set has exactly
  one queue). Any other case is sent to a human with the candidate queues and a
  reason. Combining the two signals is deliberate: the threshold controls the
  error rate on accepted tickets, while the singleton-set condition rejects
  cases the model finds genuinely ambiguous even when its top probability is
  high.
- All thresholds are fit on calibration only and frozen into
  `configs/policy.yaml`; a test asserts the calibrate path never reads the test
  split.

## Rejected options
- Scaling raw LR logits directly: ties the layer to one model type and breaks
  once the selected model changes.
- Abstaining on the threshold alone, without the conformal set: a confident but
  ambiguous prediction (two queues almost tied) would be auto-routed.
- A conformal library (MAPIE/netcal): the split-conformal rule is a few lines
  and is implemented directly (section 2.3).

## Consequence
The service and the final evaluation load the frozen `configs/policy.yaml` and
apply the same `decide(...)` function the harness uses.
