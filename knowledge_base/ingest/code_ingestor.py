from __future__ import annotations

import ast
from pathlib import Path
from typing import List

from core.schema import KnowledgeChunk
from knowledge_base.chunks.chunker import record_chunk
from knowledge_base.ingest.common import infer_model_name


def _summarize_python(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text[:1800]
    classes = []
    functions = []
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)][:12]
            classes.append(f"class {node.name}(methods={methods})")
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(text, node) or "")
    doc = ast.get_docstring(tree) or ""
    return (
        f"Docstring: {doc[:500]}\n"
        f"Imports: {'; '.join(imports[:20])}\n"
        f"Classes: {'; '.join(classes[:20])}\n"
        f"Functions: {', '.join(functions[:30])}"
    )


def _is_architecture_file(path: Path) -> bool:
    lowered = [part.lower() for part in path.parts]
    if path.suffix != ".py":
        return False
    return any(part in {"models", "model", "layers", "baselines"} for part in lowered)


def ingest(asset_dir: str | Path, chunk_config: dict | None = None) -> List[KnowledgeChunk]:
    root = Path(asset_dir)
    chunks: List[KnowledgeChunk] = []
    for path in root.rglob("*.py"):
        if not _is_architecture_file(path):
            continue
        model_name = infer_model_name(path)
        rel = path.relative_to(root)
        summary = _summarize_python(path)
        if not summary.strip():
            continue
        chunks.append(
            record_chunk(
                text=f"模型架构代码摘要：模型={model_name or '未知'}；文件={rel}。\n{summary}",
                source_type="architecture",
                source_path=rel,
                source_id="code-summary",
                metadata={"model_name": model_name},
            )
        )
    return chunks

