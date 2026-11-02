# 0003: Low-sample queues and the novel-queue holdout

## Context
The English subset has ten queues. Two decisions had to be made before
splitting: whether any queue is too small to keep as a routing target, and
which queue to hold out entirely to probe novel-queue detection (section 3.2).

English queue counts (file `aa_dataset-tickets-multi-lang-5-2-50-version.csv`,
revision `ddf1c81`): Technical Support 4737, Product Support 3073, Customer
Service 2410, IT Support 1942, Billing and Payments 1595, Returns and Exchanges
820, Service Outages and Maintenance 664, Sales and Pre-Sales 513, Human
Resources 348, General Inquiry 236.

## Decision
- **No queue is dropped.** The smallest queue (General Inquiry, 236) is well
  above the 100-sample floor, so every queue stays a routing target and no
  artificial "other" class is created.
- **Service Outages and Maintenance is the novel-queue holdout.** It is removed
  from every training split; the model never sees it, and it feeds the
  novel-queue detection probe. It is mid-volume (664), so removing it leaves the
  main task intact, and it is semantically distinct enough that "unseen queue"
  detection is a meaningful test rather than a near-synonym of a kept queue.

## Rejected options
- Holding out a tiny queue (Human Resources, General Inquiry): too few samples
  to measure detection reliably.
- Holding out a dominant queue (Technical Support): would strip too much data
  from the main task and distort the remaining class balance.

## Consequence
The routing task runs over nine queues; the tenth is reserved for the
novel-queue AUROC. The `novel_queue` name lives in `configs/default.yaml`.
