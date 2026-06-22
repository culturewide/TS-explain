from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def analyze_stationarity(values: np.ndarray, variable_names: Sequence[str]) -> Dict[str, dict]:
    report: Dict[str, dict] = {}
    for idx, name in enumerate(variable_names):
        series = np.asarray(values[:, idx], dtype=float)
        entry = {}
        try:
            from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore

            adf = adfuller(series, autolag="AIC")
            entry["adf_pvalue"] = float(adf[1])
            try:
                kp = kpss(series, regression="c", nlags="auto")
                entry["kpss_pvalue"] = float(kp[1])
            except Exception:
                entry["kpss_pvalue"] = None
            entry["stationary_hint"] = bool(entry["adf_pvalue"] < 0.05 and (entry["kpss_pvalue"] is None or entry["kpss_pvalue"] > 0.05))
        except Exception:
            half = max(2, len(series) // 2)
            first_mean = float(np.nanmean(series[:half]))
            second_mean = float(np.nanmean(series[half:]))
            scale = float(np.nanstd(series) or 1.0)
            drift_ratio = abs(second_mean - first_mean) / scale
            entry["mean_drift_ratio"] = drift_ratio
            entry["stationary_hint"] = bool(drift_ratio < 0.35)
        report[name] = entry
    return report

