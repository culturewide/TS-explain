from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from core.schema import KnowledgeChunk
from core.text import normalize_text
from knowledge_base.chunks.chunker import paragraph_chunks, record_chunk
from knowledge_base.ingest.common import clean_dataset_name, infer_model_name


def _dataset_from_text(text: str) -> str | None:
    match = re.search(r"(SMD|MSL|SMAP|PSM|SWAT|GECCO|Creditcard|CICIDS|ASD_dataset_\d+|NIPS_TS_[A-Za-z]+)", text)
    return match.group(1) if match else None


def ingest(asset_dir: str | Path, chunk_config: dict | None = None) -> List[KnowledgeChunk]:
    root = Path(asset_dir)
    cfg = (chunk_config or {}).get("hyperparameter", {})
    chunks: List[KnowledgeChunk] = []

    for path in list(root.rglob("*.json")) + list(root.rglob("*.sh")):
        if any(part in {"node_modules", ".git"} for part in path.parts):
            continue
        rel = path.relative_to(root)
        model_name = infer_model_name(path)
        if path.suffix == ".json":
            try:
                raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            dataset_name = _dataset_from_text(text)
            chunks.append(
                record_chunk(
                    text=f"超参/配置记录：模型={model_name or '未知'}；数据集={dataset_name or '未指定'}；配置={text[:1600]}",
                    source_type="hyperparameter",
                    source_path=rel,
                    source_id="json-config",
                    metadata={"model_name": model_name, "dataset_name": dataset_name},
                )
            )
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not normalize_text(text):
                continue
            dataset_name = _dataset_from_text(text) or clean_dataset_name(path.parent.name)
            chunks.extend(
                paragraph_chunks(
                    text=f"运行脚本/超参配置：模型={model_name or '未知'}；数据集={dataset_name or '未指定'}。\n{text}",
                    source_type="hyperparameter",
                    source_path=rel,
                    source_id_prefix="script",
                    metadata={"model_name": model_name, "dataset_name": dataset_name},
                    max_chars=int(cfg.get("max_chars", 900)),
                    overlap_chars=int(cfg.get("overlap_chars", 80)),
                    min_chars=int(cfg.get("min_chars", 20)),
                )
            )
    return chunks

