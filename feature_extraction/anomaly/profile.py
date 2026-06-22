from __future__ import annotations

from typing import Optional

import numpy as np

from core.schema import AnomalyProfile


def _clusters(indices: np.ndarray, gap: int) -> list[list[int]]:
    clusters: list[list[int]] = []
    for idx in indices.tolist():
        if not clusters or idx - clusters[-1][-1] > gap:
            clusters.append([int(idx)])
        else:
            clusters[-1].append(int(idx))
    return clusters


def anomaly_scores_from_window(values: np.ndarray) -> np.ndarray:
    z = np.abs((values - np.nanmean(values, axis=0)) / (np.nanstd(values, axis=0) + 1e-8))
    return np.nanmean(z, axis=1)


def analyze_anomaly_profile(
    values: np.ndarray,
    anomaly_scores: Optional[np.ndarray] = None,
    *,
    score_quantile: float = 0.95,
    cluster_gap: int = 3,
) -> AnomalyProfile:
    scores = np.asarray(anomaly_scores, dtype=float).reshape(-1) if anomaly_scores is not None else anomaly_scores_from_window(values)
    if len(scores) != values.shape[0]:
        scores = np.resize(scores, values.shape[0])
    threshold = float(np.nanquantile(scores, score_quantile)) if len(scores) else 0.0
    abnormal = np.where(scores >= threshold)[0]
    clusters = _clusters(abnormal, gap=cluster_gap) if len(abnormal) else []
    peak_position = int(np.nanargmax(scores)) if len(scores) else None
    density = float(len(abnormal) / max(len(scores), 1))
    if not len(abnormal):
        distribution = "未形成明显高分异常区域"
    elif len(clusters) == 1:
        distribution = "异常分数集中在单个连续区域"
    elif len(clusters) <= 3:
        distribution = "异常分数呈少数簇状分布"
    else:
        distribution = "异常分数较分散，可能存在多段异常或噪声"
    return AnomalyProfile(
        density=density,
        cluster_count=len(clusters),
        peak_position=peak_position,
        temporal_distribution=distribution,
        score_summary={
            "threshold": threshold,
            "max": float(np.nanmax(scores)) if len(scores) else 0.0,
            "mean": float(np.nanmean(scores)) if len(scores) else 0.0,
            "clusters": clusters,
        },
    )
