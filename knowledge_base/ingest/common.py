from __future__ import annotations

from pathlib import Path
from typing import Optional


MODEL_ALIASES = {
    "Anomaly-Transformer": "Anomaly Transformer",
    "KDD2023-DCdetector": "DCdetector",
    "DADA": "DADA",
    "CATCH": "CATCH",
    "MtsCID-main": "MtsCID",
    "TimeMixer": "TimeMixer",
    "iTransformer": "iTransformer",
}


def infer_model_name(path: str | Path) -> Optional[str]:
    parts = Path(path).parts
    for part in parts:
        if part in MODEL_ALIASES:
            return MODEL_ALIASES[part]
    return None


def clean_dataset_name(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "multi"}:
        return None
    if text.lower().endswith(".csv"):
        text = Path(text).stem
    return text

