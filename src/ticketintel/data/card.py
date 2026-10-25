from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ticketintel.config import Settings
from ticketintel.data import schema as sch

# Facts verified against the dataset card and the Hugging Face API before any
# code was written (section 3.2). Recorded here so the data card never guesses.
LICENSE = "cc-by-nc-4.0"
IS_SYNTHETIC = True


def _distribution(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    table = frame.groupby([sch.SPLIT, column]).size().unstack(fill_value=0)
    return table.sort_index()


def _to_markdown(table: pd.DataFrame) -> str:
    # Render without the optional tabulate dependency.
    headers = [str(table.index.name or "")] + [str(c) for c in table.columns]
    rows: list[list[str]] = [headers, ["---"] * len(headers)]
    for index, row in table.iterrows():
        rows.append([str(index)] + [str(v) for v in row.tolist()])
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def _quick_macro_f1(
    texts: list[str],
    labels: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> float:
    vectorizer = TfidfVectorizer(min_df=2)
    matrix = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(matrix[train_idx], np.asarray(labels)[train_idx])
    pred = model.predict(matrix[test_idx])
    return float(f1_score(np.asarray(labels)[test_idx], pred, average="macro"))


def leakage_comparison(frame: pd.DataFrame, settings: Settings) -> dict[str, float]:
    # The headline leakage finding: a random split lets near-duplicates bleed
    # across train/test and inflates macro-F1 versus a group-aware split.
    work = frame.loc[frame[sch.SPLIT] != "novel_queue"].reset_index(drop=True)
    texts = work[sch.TEXT].tolist()
    labels = work[sch.QUEUE].tolist()
    groups = work[sch.GROUP_ID].to_numpy()
    idx = np.arange(len(work))

    rand_train, rand_test = train_test_split(
        idx, test_size=0.2, random_state=settings.seed, stratify=labels
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=settings.seed)
    group_train, group_test = next(gss.split(idx, labels, groups))

    return {
        "random_split_macro_f1": _quick_macro_f1(
            texts, labels, rand_train, rand_test, settings.seed
        ),
        "group_aware_macro_f1": _quick_macro_f1(
            texts, labels, group_train, group_test, settings.seed
        ),
    }


def render(frame: pd.DataFrame, settings: Settings, raw_rows: int) -> str:
    text_len = frame[sch.TEXT].str.len()
    n_groups = frame[sch.GROUP_ID].nunique()
    duplicate_rows = len(frame) - n_groups
    leakage = leakage_comparison(frame, settings)

    lines: list[str] = []
    lines.append("# Data card\n")
    lines.append("## Source\n")
    lines.append(f"- Dataset: `{settings.dataset.id}`")
    lines.append(f"- Revision: `{settings.dataset.revision}`")
    lines.append(f"- File: `{settings.dataset.file}`")
    lines.append(f"- License: `{LICENSE}` (non-commercial; personal/educational use only)")
    lines.append(f"- Synthetic: {IS_SYNTHETIC} (generated tickets, not real customer data)")
    lines.append(f"- Language kept: `{settings.dataset.language}`\n")

    lines.append("## Row counts\n")
    lines.append(f"- English rows loaded: {raw_rows}")
    lines.append(f"- Rows after cleaning: {len(frame)}")
    lines.append(f"- Duplicate / near-duplicate groups: {n_groups}")
    lines.append(f"- Rows removed as duplicates of a representative: {duplicate_rows}")
    lines.append(f"- Near-duplicate threshold: {settings.dedup.near_duplicate_threshold}\n")

    lines.append("## Text length (characters)\n")
    lines.append(
        f"- min / median / max: {int(text_len.min())} / "
        f"{int(text_len.median())} / {int(text_len.max())}\n"
    )

    lines.append("## Queue distribution by split\n")
    lines.append(_to_markdown(_distribution(frame, sch.QUEUE)) + "\n")

    lines.append("## Priority distribution by split\n")
    lines.append(_to_markdown(_distribution(frame, sch.PRIORITY)) + "\n")

    lines.append("## Near-duplicate leakage\n")
    lines.append(
        "A fast TF-IDF baseline on a random split versus a group-aware split. "
        "The gap is the leakage a random split would hide.\n"
    )
    lines.append(f"- Random split macro-F1: {leakage['random_split_macro_f1']:.4f}")
    lines.append(f"- Group-aware macro-F1: {leakage['group_aware_macro_f1']:.4f}")
    lines.append(
        f"- Difference: {leakage['random_split_macro_f1'] - leakage['group_aware_macro_f1']:.4f}\n"
    )

    lines.append("## Held-out novel queue\n")
    lines.append(f"- `{settings.novel_queue}` is removed from all training splits (ADR 0003).\n")

    return "\n".join(lines)


def write(frame: pd.DataFrame, settings: Settings, raw_rows: int) -> Path:
    content = render(frame, settings, raw_rows)
    out_dir = settings.paths.reports
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data_card.md"
    path.write_text(content, encoding="utf-8")
    return path
