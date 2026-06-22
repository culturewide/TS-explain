from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from core.schema import KnowledgeChunk
from knowledge_base.chunks.chunker import record_chunk
from knowledge_base.ingest.common import clean_dataset_name


def ingest(asset_dir: str | Path, chunk_config: dict | None = None) -> List[KnowledgeChunk]:
    root = Path(asset_dir)
    chunks: List[KnowledgeChunk] = []
    for path in root.rglob("DETECT_META.csv"):
        try:
            df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
        except Exception:
            continue
        rel = path.relative_to(root)
        for idx, row in df.iterrows():
            dataset_name = clean_dataset_name(row.get("file_name")) or clean_dataset_name(row.get("dataset_name"))
            fields = [f"{key}={value}" for key, value in row.items() if not pd.isna(value)]
            text = f"数据集元数据：数据集={dataset_name or '未知'}；" + "；".join(fields)
            chunks.append(
                record_chunk(
                    text=text,
                    source_type="dataset_meta",
                    source_path=rel,
                    source_id=f"meta:{idx + 1}",
                    metadata={"dataset_name": dataset_name},
                )
            )
    return chunks

