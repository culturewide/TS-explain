from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core.config import load_yaml, project_root
from core.schema import ResidualFact, TSFactCard
from feature_extraction.anomaly.attribution import variable_attribution
from feature_extraction.anomaly.profile import analyze_anomaly_profile
from feature_extraction.correlation.inter_variable import analyze_correlations
from feature_extraction.prompt_builder import build_fact_card_text
from feature_extraction.statistical.changepoint import detect_changepoints
from feature_extraction.statistical.periodicity import analyze_periodicity
from feature_extraction.statistical.stationarity import analyze_stationarity
from feature_extraction.statistical.trend import analyze_trend


def _to_numpy(window: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> tuple[np.ndarray, list[str]]:
    if isinstance(window, pd.DataFrame):
        return window.to_numpy(dtype=float), [str(col) for col in window.columns]
    arr = np.asarray(window, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    names = [f"var_{idx + 1}" for idx in range(arr.shape[1])]
    return arr, names


def _residual_facts(residuals: Optional[np.ndarray], variable_names: Sequence[str], z_threshold: float) -> list[ResidualFact]:
    if residuals is None:
        return []
    res = np.asarray(residuals, dtype=float)
    if res.ndim == 1:
        res = res.reshape(-1, 1)
    facts: list[ResidualFact] = []
    for idx, name in enumerate(variable_names[: res.shape[1]]):
        series = res[:, idx]
        z = np.abs((series - np.nanmean(series)) / (np.nanstd(series) + 1e-8))
        regions = [str(int(pos)) for pos in np.where(z >= z_threshold)[0][:10]]
        facts.append(
            ResidualFact(
                variable=name,
                mean_bias=float(np.nanmean(series)),
                std=float(np.nanstd(series)),
                max_abs_error=float(np.nanmax(np.abs(series))),
                amplification_regions=regions,
            )
        )
    return facts


class TSFeatureExtractor:
    def __init__(self, config_path: str | Path | None = None):
        self.config = load_yaml(config_path or (project_root() / "config" / "feature_config.yaml"))

    def extract(
        self,
        window: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
        *,
        window_id: str = "window-unknown",
        anomaly_scores: Optional[Sequence[float]] = None,
        residuals: Optional[pd.DataFrame | np.ndarray | Sequence[Sequence[float]]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> TSFactCard:
        values, inferred_names = _to_numpy(window)
        variable_names = list(feature_names or inferred_names)
        if len(variable_names) != values.shape[1]:
            variable_names = [f"var_{idx + 1}" for idx in range(values.shape[1])]
        scores = np.asarray(anomaly_scores, dtype=float) if anomaly_scores is not None else None
        residual_arr = None
        if residuals is not None:
            residual_arr, _ = _to_numpy(residuals)  # type: ignore[arg-type]

        trend_cfg = self.config.get("trend", {})
        periodic_cfg = self.config.get("periodicity", {})
        anomaly_cfg = self.config.get("anomaly", {})
        corr_cfg = self.config.get("correlation", {})
        residual_cfg = self.config.get("residual", {})
        cp_cfg = self.config.get("changepoint", {})

        card = TSFactCard(
            window_id=window_id,
            length=values.shape[0],
            variables=variable_names,
            trends=analyze_trend(
                values,
                variable_names,
                stl_period=trend_cfg.get("stl_period"),
                moderate_threshold=float(trend_cfg.get("kendall_moderate_threshold", 0.20)),
                strong_threshold=float(trend_cfg.get("kendall_strong_threshold", 0.45)),
            ),
            periodicities=analyze_periodicity(
                values,
                variable_names,
                max_lag=int(periodic_cfg.get("max_lag", 80)),
                acf_threshold=float(periodic_cfg.get("acf_threshold", 0.35)),
                fft_power_ratio_threshold=float(periodic_cfg.get("fft_power_ratio_threshold", 0.18)),
            ),
            correlations=analyze_correlations(
                values,
                variable_names,
                top_pairs=int(corr_cfg.get("top_pairs", 6)),
                change_threshold=float(corr_cfg.get("change_threshold", 0.35)),
            ),
            anomaly_profile=analyze_anomaly_profile(
                values,
                scores,
                score_quantile=float(anomaly_cfg.get("score_quantile", 0.95)),
                cluster_gap=int(anomaly_cfg.get("cluster_gap", 3)),
            ),
            residuals=_residual_facts(
                residual_arr,
                variable_names,
                z_threshold=float(residual_cfg.get("amplification_z", 2.0)),
            ),
            attributions=variable_attribution(values, variable_names, residual_arr, scores),
            changepoints=detect_changepoints(
                values,
                max_points=int(cp_cfg.get("max_points", 5)),
                penalty=float(cp_cfg.get("penalty", 3.0)),
            ),
            stationarity=analyze_stationarity(values, variable_names),
        )
        card.narrative = build_fact_card_text(card)
        return card
