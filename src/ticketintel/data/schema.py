from __future__ import annotations

from typing import Final

# Source columns kept from the dataset. The dataset also carries answer, type,
# version and tag_* columns that the routing/summarization tasks do not use.
SUBJECT: Final = "subject"
BODY: Final = "body"
QUEUE: Final = "queue"
PRIORITY: Final = "priority"
LANGUAGE: Final = "language"

# Derived columns this pipeline adds.
TEXT: Final = "text"  # model input: subject + "\n" + body
GROUP_ID: Final = "group_id"  # near-duplicate group assigned in dedup
SPLIT: Final = "split"  # train / validation / calibration / test / novel_queue

SOURCE_COLUMNS: Final = (SUBJECT, BODY, QUEUE, PRIORITY, LANGUAGE)

# The three ordinal priority levels, lowest to highest (section 3.3, task B).
PRIORITY_ORDER: Final = ("low", "medium", "high")
