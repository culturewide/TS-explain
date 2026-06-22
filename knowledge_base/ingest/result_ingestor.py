from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from core.schema import KnowledgeChunk
from knowledge_base.chunks.chunker import record_chunk
from knowledge_base.ingest.common import clean_dataset_name, infer_model_name


SUMMARY_PATTERNS = (
    "results_summary.csv",
    "dc_detector_results.csv",
    "MtsLINE_benchmark_result.csv",
    "anomaly_transformer_ultimate_results.csv",
    "anomaly_transformer_ultimate_results2.csv",
    "_auc.csv",
    "_affiliation.csv",
)


def _is_result_file(path: Path) -> bool:
    name = path.name
    if name.startswith("test_report") and name.endswith(".csv"):
        return True
    return any(name.endswith(pattern) or name == pattern for pattern in SUMMARY_PATTERNS)


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    except Exception:
        try:
            return pd.read_csv(path, encoding="gbk", encoding_errors="replace")
        except Exception:
            return None


def _infer_dataset_from_path(path: Path, row: pd.Series) -> Optional[str]:
    for key in ("Dataset", "dataset", "dataset_name", "file_name"):
        if key in row:
            dataset = clean_dataset_name(row[key])
            if dataset:
                return dataset
    parts = list(path.parts)
    if "_auc.csv" in path.name or "_affiliation.csv" in path.name:
        if len(parts) >= 2:
            return clean_dataset_name(parts[-2])
    return None


def _metric_text(row: pd.Series, max_items: int = 20) -> str:
    pairs = []
    for key, value in row.items():
        if pd.isna(value):
            continue
        text = str(value)
        if len(text) > 160:
            text = text[:157] + "..."
        pairs.append(f"{key}={text}")
        if len(pairs) >= max_items:
            break
    return "; ".join(pairs)


def _rows_for_file(path: Path, df: pd.DataFrame) -> Iterable[pd.Series]:
    if path.name == "_affiliation.csv" and "Affiliation_F1" in df.columns:
        ranked = df.sort_values("Affiliation_F1", ascending=False).head(3)
        for _, row in ranked.iterrows():
            yield row
        return
    if len(df) > 80 and "metric_name" not in df.columns:
        for _, row in df.head(80).iterrows():
            yield row
        return
    for _, row in df.iterrows():
        yield row


def ingest(asset_dir: str | Path, chunk_config: dict | None = None) -> List[KnowledgeChunk]:
    root = Path(asset_dir)
    chunks: List[KnowledgeChunk] = []
    for path in root.rglob("*.csv"):
        if not _is_result_file(path):
            continue
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        model_name = infer_model_name(path)
        rel = path.relative_to(root)
        for idx, row in enumerate(_rows_for_file(path, df), start=1):
            dataset_name = _infer_dataset_from_path(path, row)
            metrics = _metric_text(row)
            text = (
                f"实验结果记录：模型={model_name or '未知'}；数据集={dataset_name or '未知'}；"
                f"来源={rel}；指标/参数：{metrics}"
            )
            metadata = {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "file_name": path.name,
                "row_index": idx,
            }
            chunks.append(
                record_chunk(
                    text=text,
                    source_type="experiment_result",
                    source_path=rel,
                    source_id=f"row:{idx}",
                    metadata=metadata,
                )
            )
    return chunks

