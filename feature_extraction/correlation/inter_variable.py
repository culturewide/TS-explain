from __future__ import annotations

from itertools import combinations
from typing import List, Sequence

import numpy as np
from scipy.stats import spearmanr

from core.schema import CorrelationFact


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def analyze_correlations(
    values: np.ndarray,
    variable_names: Sequence[str],
    *,
    top_pairs: int = 6,
    change_threshold: float = 0.35,
) -> List[CorrelationFact]:
    facts: List[CorrelationFact] = []
    n = values.shape[0]
    half = max(2, n // 2)
    for i, j in combinations(range(values.shape[1]), 2):
        a = np.asarray(values[:, i], dtype=float)
        b = np.asarray(values[:, j], dtype=float)
        pearson = _safe_corr(a, b)
        if np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
            sp = None
        else:
            try:
                sp = float(spearmanr(a, b, nan_policy="omit").statistic or 0.0)
            except Exception:
                sp = None
        before = _safe_corr(a[:half], b[:half])
        after = _safe_corr(a[half:], b[half:])
        facts.append(
            CorrelationFact(
                variable_a=variable_names[i],
                variable_b=variable_names[j],
                pearson=pearson,
                spearman=sp,
                changed=bool(abs(after - before) >= change_threshold),
            )
        )
    facts.sort(key=lambda item: abs(item.pearson), reverse=True)
    return facts[:top_pairs]
