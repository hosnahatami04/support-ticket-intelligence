from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ticketintel.data import schema
from ticketintel.logging import get_logger

logger = get_logger(__name__)

_WS = re.compile(r"\s+")


def _exact_key(text: str) -> str:
    normalized = _WS.sub(" ", text.lower()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _near_duplicate_edges(texts: list[str], threshold: float) -> list[tuple[int, int]]:
    # TF-IDF + cosine nearest neighbours. Two tickets whose similarity exceeds
    # the threshold become an edge; connected components are the groups.
    vectorizer = TfidfVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(texts)
    radius = 1.0 - threshold
    nn = NearestNeighbors(radius=radius, metric="cosine")
    nn.fit(matrix)
    distances, indices = nn.radius_neighbors(matrix, return_distance=True)

    edges: list[tuple[int, int]] = []
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices, strict=True)):
        for dist, j in zip(dist_row, idx_row, strict=True):
            if j > i and (1.0 - dist) >= threshold:
                edges.append((i, j))
    return edges


def _components(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    if edges:
        rows = np.array([e[0] for e in edges])
        cols = np.array([e[1] for e in edges])
    else:
        rows = np.empty(0, dtype=np.intp)
        cols = np.empty(0, dtype=np.intp)
    data = np.ones(len(edges), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return np.asarray(labels)


def assign_groups(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = frame.copy()
    texts = out[schema.TEXT].tolist()

    # Exact duplicates first: collapse identical normalized text to one
    # representative so near-duplicate search runs on unique content only.
    exact_keys = [_exact_key(t) for t in texts]
    representative: dict[str, int] = {}
    for position, key in enumerate(exact_keys):
        representative.setdefault(key, position)
    unique_positions = sorted(set(representative.values()))

    unique_texts = [texts[p] for p in unique_positions]
    edges = _near_duplicate_edges(unique_texts, threshold)
    unique_labels = _components(len(unique_positions), edges)

    position_to_group = {
        pos: int(label) for pos, label in zip(unique_positions, unique_labels, strict=True)
    }
    groups = [position_to_group[representative[key]] for key in exact_keys]
    out[schema.GROUP_ID] = groups

    n_groups = len(set(groups))
    logger.info(
        "assigned duplicate groups",
        extra={"rows": len(out), "groups": n_groups, "near_dup_edges": len(edges)},
    )
    return out
