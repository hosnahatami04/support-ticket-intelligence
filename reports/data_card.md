# Data card

## Source

- Dataset: `Tobi-Bueck/customer-support-tickets`
- Revision: `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`
- File: `aa_dataset-tickets-multi-lang-5-2-50-version.csv`
- License: `cc-by-nc-4.0` (non-commercial; personal/educational use only)
- Synthetic: True (generated tickets, not real customer data)
- Language kept: `en`

## Row counts

- English rows loaded: 16338
- Rows after cleaning: 16338
- Duplicate / near-duplicate groups: 14664
- Rows removed as duplicates of a representative: 1674
- Near-duplicate threshold: 0.9

## Text length (characters)

- min / median / max: 22 / 417 / 1183

## Queue distribution by split

| split | Billing and Payments | Customer Service | General Inquiry | Human Resources | IT Support | Product Support | Returns and Exchanges | Sales and Pre-Sales | Service Outages and Maintenance | Technical Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | 159 | 241 | 23 | 35 | 194 | 307 | 82 | 51 | 0 | 473 |
| novel_queue | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 664 | 0 |
| test | 159 | 241 | 24 | 35 | 194 | 307 | 82 | 51 | 0 | 474 |
| train | 1116 | 1687 | 165 | 243 | 1360 | 2152 | 574 | 359 | 0 | 3316 |
| validation | 160 | 241 | 24 | 35 | 194 | 307 | 82 | 52 | 0 | 474 |

## Priority distribution by split

| split | high | low | medium |
| --- | --- | --- | --- |
| calibration | 614 | 321 | 630 |
| novel_queue | 467 | 83 | 115 |
| test | 614 | 309 | 644 |
| train | 4100 | 2319 | 4553 |
| validation | 551 | 342 | 676 |

## Near-duplicate leakage

A fast TF-IDF baseline on a random split versus a group-aware split. The gap is the leakage a random split would hide.

- Random split macro-F1: 0.4092
- Group-aware macro-F1: 0.3843
- Difference: 0.0249

## Held-out novel queue

- `Service Outages and Maintenance` is removed from all training splits (ADR 0003).
