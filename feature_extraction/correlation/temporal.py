from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def lagged_dependency(values: np.ndarray, variable_names: Sequence[str], max_lag: int = 5) -> Dict[str, dict]:
    output: Dict[str, dict] = {}
    for idx, name in enumerate(variable_names):
        series = np.asarray(values[:, idx], dtype=float)
        best_lag = 0
        best_corr = 0.0
        for lag in range(1, min(max_lag, len(series) - 2) + 1):
            if np.nanstd(series[:-lag]) <= 1e-12 or np.nanstd(series[lag:]) <= 1e-12:
                continue
            corr = float(np.corrcoef(series[:-lag], series[lag:])[0, 1])
            if abs(corr) > abs(best_corr):
                best_lag = lag
                best_corr = corr
        output[name] = {"best_lag": best_lag, "lagged_corr": best_corr}
    return output

