from __future__ import annotations

from typing import List

import numpy as np


def detect_changepoints(values: np.ndarray, *, max_points: int = 5, penalty: float = 3.0) -> List[int]:
    if values.shape[0] < 8:
        return []
    signal = np.nanmean(values, axis=1)
    try:
        import ruptures as rpt  # type: ignore

        algo = rpt.Pelt(model="rbf").fit(signal)
        points = algo.predict(pen=penalty)
        return [int(p) for p in points if 0 < p < len(signal)][:max_points]
    except Exception:
        scores = []
        for idx in range(3, len(signal) - 3):
            left = signal[:idx]
            right = signal[idx:]
            pooled = float(np.nanstd(signal) or 1.0)
            score = abs(float(np.nanmean(left) - np.nanmean(right))) / pooled
            scores.append((idx, score))
        selected = [idx for idx, score in sorted(scores, key=lambda item: item[1], reverse=True) if score > 0.6]
        selected = sorted(selected[:max_points])
        compact: List[int] = []
        for point in selected:
            if not compact or point - compact[-1] > 3:
                compact.append(point)
        return compact

