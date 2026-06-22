from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from core.config import load_project_config, resolve_path
from core.schema import TSFactCard
from feature_extraction.extractor import TSFeatureExtractor


WindowStrategy = Literal["max_label_density", "first_anomaly"]


@dataclass(slots=True)
class RealWindow:
    """A real dataset window resolved from local assets."""

    dataset_name: str
    window_id: str
    start: int
    end: int
    values: pd.DataFrame
    labels: np.ndarray
    label_density: float
    source_path: str
    label_path: str
    strategy: str
    note: str


class RealWindowResolver:
    """Resolve real time-series windows from known local datasets.

    Current first adapter: SMD from Anomaly-Transformer/dataset/SMD.
    The window is selected by ground-truth label density because DADA's local
    assets contain aggregate metrics but no unified point-level score file.
    """

    def __init__(self, config_path: str | Path | None = None):
        self.config = load_project_config(config_path)
        root = Path(self.config["_project_root"])
        self.assets_dir = resolve_path(self.config["project"]["assets_dir"], root)

    def resolve(
        self,
        dataset_name: str,
        *,
        window_length: int = 256,
        strategy: WindowStrategy = "max_label_density",
    ) -> RealWindow:
        normalized = dataset_name.strip().lower()
        if normalized != "smd":
            raise ValueError("当前真实窗口自动解析只支持 SMD；其他数据集需要再添加适配器。")
        return self._resolve_smd(window_length=window_length, strategy=strategy)

    def extract_fact_card(
        self,
        dataset_name: str,
        *,
        model_name: Optional[str] = None,
        window_length: int = 256,
        strategy: WindowStrategy = "max_label_density",
    ) -> tuple[RealWindow, TSFactCard]:
        real_window = self.resolve(dataset_name, window_length=window_length, strategy=strategy)
        extractor = TSFeatureExtractor()
        card = extractor.extract(
            real_window.values,
            window_id=real_window.window_id,
            anomaly_scores=real_window.labels.astype(float),
            feature_names=list(real_window.values.columns),
        )
        model_hint = f"；问题模型={model_name}" if model_name else ""
        source_note = (
            f"真实窗口来源：{real_window.source_path}[{real_window.start}:{real_window.end}]；"
            f"异常定位信号来自 {real_window.label_path}，标签密度 {real_window.label_density:.2%}{model_hint}。"
            "注意：当前窗口使用真实标签定位，不代表该模型的逐点 anomaly score。"
        )
        card.narrative = f"{source_note}\n{card.narrative}"
        return real_window, card

    def _resolve_smd(self, *, window_length: int, strategy: WindowStrategy) -> RealWindow:
        smd_dir = self.assets_dir / "Anomaly-Transformer" / "dataset" / "SMD"
        test_path = smd_dir / "SMD_test.npy"
        label_path = smd_dir / "SMD_test_label.npy"
        if not test_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"未找到 SMD 测试序列或标签：{smd_dir}")

        values = np.load(test_path, mmap_mode="r")
        labels = np.asarray(np.load(label_path, mmap_mode="r"), dtype=float).reshape(-1)
        if values.shape[0] != labels.shape[0]:
            raise ValueError(f"SMD 序列长度和标签长度不一致：{values.shape[0]} vs {labels.shape[0]}")

        length = int(max(8, min(window_length, values.shape[0])))
        start = self._select_start(labels, length, strategy)
        end = min(start + length, values.shape[0])
        start = max(0, end - length)

        window_values = np.asarray(values[start:end], dtype=float)
        window_labels = labels[start:end].astype(float)
        columns = [f"sensor_{idx + 1:02d}" for idx in range(window_values.shape[1])]
        frame = pd.DataFrame(window_values, columns=columns)
        density = float(np.mean(window_labels)) if len(window_labels) else 0.0
        window_id = f"SMD-real-{strategy}-{start}-{end}"
        note = (
            "SMD 真实测试窗口；窗口由 test_label.npy 选择，"
            "适合解释该真实异常片段的时序形态。"
        )
        return RealWindow(
            dataset_name="SMD",
            window_id=window_id,
            start=start,
            end=end,
            values=frame,
            labels=window_labels,
            label_density=density,
            source_path=str(test_path),
            label_path=str(label_path),
            strategy=strategy,
            note=note,
        )

    @staticmethod
    def _select_start(labels: np.ndarray, window_length: int, strategy: WindowStrategy) -> int:
        if len(labels) <= window_length:
            return 0
        positive = np.flatnonzero(labels > 0)
        if len(positive) == 0:
            return 0
        if strategy == "first_anomaly":
            return int(max(0, min(positive[0] - window_length // 4, len(labels) - window_length)))
        if strategy != "max_label_density":
            raise ValueError(f"未知窗口选择策略：{strategy}")
        counts = np.convolve(labels.astype(float), np.ones(window_length, dtype=float), mode="valid")
        return int(np.argmax(counts))
