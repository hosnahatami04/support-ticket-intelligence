from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from ticketintel.config import Settings
from ticketintel.data import schema
from ticketintel.evaluation import metrics
from ticketintel.evaluation.final import FrozenPolicy, load_split
from ticketintel.logging import get_logger
from ticketintel.models.tfidf import TfidfClassifier
from ticketintel.reliability import calibration, selective
from ticketintel.reliability.policy import AUTO_ROUTE, decide

logger = get_logger(__name__)


def _word_only_pipeline(c: float = 30.0) -> Pipeline:
    return Pipeline(
        [
            ("features", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)),
            (
                "clf",
                LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=0),
            ),
        ]
    )


def _word_char_pipeline(c: float = 30.0) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=0),
            ),
        ]
    )


def ablation_split_strategy(settings: Settings) -> dict[str, float]:
    # How much a random split would have inflated the headline metric.
    frame = pd.read_parquet(settings.paths.data_processed / "tickets.parquet")
    work = frame.loc[frame[schema.SPLIT] != "novel_queue"].reset_index(drop=True)
    texts = work[schema.TEXT].tolist()
    labels = work[schema.QUEUE].tolist()
    groups = work[schema.GROUP_ID].to_numpy()
    idx = np.arange(len(work))

    def score(train_idx: np.ndarray, test_idx: np.ndarray) -> float:
        pipe = _word_char_pipeline()
        pipe.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
        pred = pipe.predict([texts[i] for i in test_idx])
        return metrics.macro_f1([labels[i] for i in test_idx], list(pred))

    rand_tr, rand_te = train_test_split(
        idx, test_size=0.2, random_state=settings.seed, stratify=labels
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=settings.seed)
    grp_tr, grp_te = next(gss.split(idx, labels, groups))

    random_f1 = score(rand_tr, rand_te)
    group_f1 = score(grp_tr, grp_te)
    return {
        "random_split_macro_f1": random_f1,
        "group_aware_macro_f1": group_f1,
        "inflation": random_f1 - group_f1,
    }


def ablation_features(settings: Settings) -> dict[str, float]:
    train = load_split(settings, "train")
    val = load_split(settings, "validation")
    out: dict[str, float] = {}
    for name, builder in (
        ("word_only", _word_only_pipeline),
        ("word_plus_char", _word_char_pipeline),
    ):
        pipe = builder()
        pipe.fit(train[schema.TEXT].tolist(), train[schema.QUEUE].tolist())
        pred = pipe.predict(val[schema.TEXT].tolist())
        out[name] = metrics.macro_f1(val[schema.QUEUE].tolist(), list(pred))
    out["char_gain"] = out["word_plus_char"] - out["word_only"]
    return out


def ablation_calibration_and_policy(settings: Settings, policy: FrozenPolicy) -> dict[str, float]:
    # Two ablations share a fit: temperature on/off, and threshold with or
    # without the conformal singleton condition.
    train = load_split(settings, "train")
    test = load_split(settings, "test")
    model = TfidfClassifier(c=30.0).fit(train[schema.TEXT].tolist(), train[schema.QUEUE].tolist())
    classes = model.classes_
    y_true = test[schema.QUEUE].tolist()
    raw = model.predict_proba(test[schema.TEXT].tolist())
    scaled = calibration.apply_temperature(raw, policy.temperature)

    y_pred = [str(classes[i]) for i in raw.argmax(axis=1)]
    correct = np.array([int(p == t) for p, t in zip(y_pred, y_true, strict=True)])

    out = {
        "ece_without_scaling": calibration.expected_calibration_error(raw.max(axis=1), correct),
        "ece_with_scaling": calibration.expected_calibration_error(scaled.max(axis=1), correct),
        "aurc_without_scaling": selective.area_under_risk_coverage(raw.max(axis=1), correct),
        "aurc_with_scaling": selective.area_under_risk_coverage(scaled.max(axis=1), correct),
    }

    threshold_only = scaled.max(axis=1) >= policy.abstention_threshold
    combined = np.array(
        [
            decide(row, classes, policy.abstention_threshold, policy.conformal_qhat).action
            == AUTO_ROUTE
            for row in scaled
        ]
    )
    for name, mask in (("threshold_only", threshold_only), ("threshold_plus_conformal", combined)):
        count = int(mask.sum())
        out[f"{name}_coverage"] = count / len(test)
        out[f"{name}_error"] = (
            float(((~correct.astype(bool)) & mask).sum() / count) if count else 0.0
        )
    return out


def run_all(settings: Settings, policy: FrozenPolicy) -> dict[str, object]:
    return {
        "split_strategy": ablation_split_strategy(settings),
        "features": ablation_features(settings),
        "calibration_and_policy": ablation_calibration_and_policy(settings, policy),
    }
