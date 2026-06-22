from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from core.schema import KnowledgeChunk
from knowledge_base.chunks.chunker import paragraph_chunks, record_chunk
from knowledge_base.ingest.common import MODEL_ALIASES, infer_model_name


CURATED_MODEL_DESCRIPTIONS: Dict[str, str] = {
    "Anomaly Transformer": (
        "Anomaly Transformer models point associations and series associations, then uses the association "
        "discrepancy between local prior associations and learned temporal associations as anomaly evidence. "
        "It is useful for explaining windows where attention behavior deviates from normal temporal dependence."
    ),
    "DCdetector": (
        "DCdetector is a dual attention contrastive anomaly detector. It compares patch-wise representations "
        "under different receptive fields and uses representation discrepancy to identify anomalous temporal segments."
    ),
    "DADA": (
        "DADA is designed as a general time-series anomaly detector with adaptive bottlenecks and dual adversarial "
        "decoders. It targets cross-domain use, emphasizing one-model-for-many deployment and zero/few adaptation."
    ),
    "CATCH": (
        "CATCH is a channel-aware multivariate anomaly detector based on frequency patching. Its channel fusion "
        "module learns patch-wise channel relationships and separates useful channel correlations from irrelevant ones."
    ),
    "MtsCID": (
        "MtsCID/MtsLINE-style architectures combine multiscale temporal patching, channel interaction modules, "
        "and reconstruction/forecasting objectives to capture variable-level and temporal dependencies."
    ),
    "TimeMixer": (
        "TimeMixer decomposes time series into multiscale temporal components and mixes past/future information "
        "across scales, which is useful when anomaly behavior is tied to trend or seasonal components."
    ),
    "iTransformer": (
        "iTransformer treats variates as tokens and applies attention over variable dimensions, making it strong "
        "for multivariate forecasting and useful as a baseline for variable-dependence explanations."
    ),
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # type: ignore

        doc = fitz.open(path)
        return "\n\n".join(page.get_text() for page in doc)
    except Exception:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""


def ingest(asset_dir: str | Path, chunk_config: dict | None = None) -> List[KnowledgeChunk]:
    root = Path(asset_dir)
    cfg = (chunk_config or {}).get("paper", {})
    chunks: List[KnowledgeChunk] = []

    for model_name, description in CURATED_MODEL_DESCRIPTIONS.items():
        chunks.append(
            record_chunk(
                text=f"模型论文/架构摘要：{model_name}. {description}",
                source_type="paper_summary",
                source_path=f"curated/{model_name}",
                source_id="curated-summary",
                metadata={"model_name": model_name, "language": "zh-en"},
            )
        )

    for readme in [path for path in root.rglob("*.md") if path.name.lower() == "readme.md"]:
        model_name = infer_model_name(readme)
        text = _read_text(readme)
        if not text.strip():
            continue
        chunks.extend(
            paragraph_chunks(
                text=text,
                source_type="paper_summary",
                source_path=readme.relative_to(root),
                source_id_prefix="readme",
                metadata={"model_name": model_name, "language": "en"},
                max_chars=int(cfg.get("max_chars", 1200)),
                overlap_chars=int(cfg.get("overlap_chars", 160)),
                min_chars=int(cfg.get("min_chars", 100)),
            )
        )

    for pdf in root.rglob("*.pdf"):
        text = _read_pdf(pdf)
        if not text.strip():
            continue
        chunks.extend(
            paragraph_chunks(
                text=text,
                source_type="paper_summary",
                source_path=pdf.relative_to(root),
                source_id_prefix="pdf",
                metadata={"model_name": infer_model_name(pdf), "language": "en"},
                max_chars=int(cfg.get("max_chars", 1200)),
                overlap_chars=int(cfg.get("overlap_chars", 160)),
                min_chars=int(cfg.get("min_chars", 100)),
            )
        )
    return chunks
