from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.stats import kendalltau

from core.schema import TrendFact


def _strength(abs_tau: float, moderate: float, strong: float) -> str:
    if abs_tau >= strong:
        return "强"
    if abs_tau >= moderate:
        return "中等"
    if abs_tau >= 0.08:
        return "弱"
    return "不明显"


def _direction(tau: float) -> str:
    if tau > 0.08:
        return "上升"
    if tau < -0.08:
        return "下降"
    return "平稳/无单调方向"


def _stl_delta(series: np.ndarray, period: int | None) -> float | None:
    if len(series) < 6:
        return None
    try:
        from statsmodels.tsa.seasonal import STL  # type: ignore

        inferred_period = period or max(2, min(len(series) // 2, 24))
        if len(series) >= inferred_period * 2:
            trend = STL(series, period=inferred_period, robust=True).fit().trend
            return float(trend[-1] - trend[0])
    except Exception:
        pass
    win = max(3, min(len(series) // 4, 12))
    kernel = np.ones(win) / win
    smooth = np.convolve(series, kernel, mode="valid")
    return float(smooth[-1] - smooth[0]) if len(smooth) else None


def analyze_trend(
    values: np.ndarray,
    variable_names: Sequence[str],
    *,
    stl_period: int | None = None,
    moderate_threshold: float = 0.20,
    strong_threshold: float = 0.45,
) -> List[TrendFact]:
    facts: List[TrendFact] = []
    x = np.arange(values.shape[0])
    for idx, name in enumerate(variable_names):
        series = np.asarray(values[:, idx], dtype=float)
        if np.allclose(series, series[0]):
            tau = 0.0
        else:
            tau = float(kendalltau(x, series, nan_policy="omit").statistic or 0.0)
        facts.append(
            TrendFact(
                variable=name,
                tau=tau,
                direction=_direction(tau),
                strength=_strength(abs(tau), moderate_threshold, strong_threshold),
                stl_trend_delta=_stl_delta(series, stl_period),
            )
        )
    return facts

