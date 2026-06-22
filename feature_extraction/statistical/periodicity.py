from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.signal import find_peaks

from core.schema import PeriodicityFact


def _acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    centered = series - np.nanmean(series)
    denom = np.nansum(centered**2)
    if denom <= 1e-12:
        return np.zeros(max_lag + 1)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        acf.append(float(np.nansum(centered[:-lag] * centered[lag:]) / denom))
    return np.asarray(acf)


def _fft_period(series: np.ndarray) -> tuple[float | None, float]:
    n = len(series)
    if n < 4 or np.allclose(series, series[0]):
        return None, 0.0
    centered = series - np.nanmean(series)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if len(power) <= 1 or float(np.sum(power[1:])) <= 1e-12:
        return None, 0.0
    nonzero = power[1:]
    best = int(np.argmax(nonzero)) + 1
    ratio = float(power[best] / np.sum(nonzero))
    period = float(n / best) if best > 0 else None
    return period, ratio


def analyze_periodicity(
    values: np.ndarray,
    variable_names: Sequence[str],
    *,
    max_lag: int = 80,
    acf_threshold: float = 0.35,
    fft_power_ratio_threshold: float = 0.18,
) -> List[PeriodicityFact]:
    facts: List[PeriodicityFact] = []
    max_lag = max(1, min(max_lag, values.shape[0] - 2))
    for idx, name in enumerate(variable_names):
        series = np.asarray(values[:, idx], dtype=float)
        acf = _acf(series, max_lag)
        peaks, props = find_peaks(acf[1:], height=acf_threshold)
        peak_lag = int(peaks[0] + 1) if len(peaks) else None
        fft_period, ratio = _fft_period(series)
        significant = bool((peak_lag is not None) or ratio >= fft_power_ratio_threshold)
        facts.append(
            PeriodicityFact(
                variable=name,
                dominant_period=fft_period,
                fft_power_ratio=ratio,
                acf_peak_lag=peak_lag,
                significant=significant,
            )
        )
    return facts

