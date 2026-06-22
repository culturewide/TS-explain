from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from core.schema import AttributionFact


def variable_attribution(
    values: np.ndarray,
    variable_names: Sequence[str],
    residuals: Optional[np.ndarray] = None,
    anomaly_scores: Optional[np.ndarray] = None,
) -> List[AttributionFact]:
    signal = np.asarray(residuals, dtype=float) if residuals is not None else values
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    z = np.abs((signal - np.nanmean(signal, axis=0)) / (np.nanstd(signal, axis=0) + 1e-8))
    if anomaly_scores is not None and len(anomaly_scores) == z.shape[0]:
        weights = np.asarray(anomaly_scores, dtype=float).reshape(-1, 1)
        contribution = np.nanmean(z * (weights / (np.nanmean(weights) + 1e-8)), axis=0)
    else:
        contribution = np.nanmean(z, axis=0)
    total = float(np.nansum(contribution)) or 1.0
    facts: List[AttributionFact] = []
    for idx, raw in enumerate(contribution[: len(variable_names)]):
        share = float(raw / total)
        if residuals is not None:
            reason = "残差幅度在该窗口内相对更高"
        else:
            reason = "标准化偏离幅度在该窗口内相对更高"
        facts.append(AttributionFact(variable=variable_names[idx], contribution=share, reason=reason))
    facts.sort(key=lambda item: item.contribution, reverse=True)
    return facts

