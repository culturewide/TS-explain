from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import load_project_config, resolve_path


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _summarize_mode(results_root: Path, mode: str) -> dict:
    path = results_root / mode / "answers.jsonl"
    records = _load_records(path)
    valid = [r for r in records if r.get("metrics") and "hallucination" in r["metrics"]]
    avg_h = float(np.mean([r["metrics"]["hallucination"]["hallucination_rate"] for r in valid])) if valid else 0.0
    avg_q = float(np.mean([r["metrics"]["answer_quality"]["score"] for r in valid])) if valid else 0.0
    avg_c = float(np.mean([r["metrics"].get("citation_consistency_rate", 0.0) for r in valid])) if valid else 0.0
    return {
        "question_count": len(records),
        "valid_question_count": len(valid),
        "error_count": len(records) - len(valid),
        "avg_hallucination_rate": round(avg_h, 4),
        "avg_hallucination_pct": round(avg_h * 100, 2),
        "avg_citation_consistency_rate": round(avg_c, 4),
        "avg_citation_consistency_pct": round(avg_c * 100, 2),
        "avg_quality_score": round(avg_q, 4),
        "path": str(path),
    }


def summarize_existing_results(
    *,
    config_path: str | Path | None = None,
    modes: Sequence[str] = ("no_rag", "rag_ts"),
) -> dict:
    config = load_project_config(config_path)
    root = Path(config["_project_root"])
    results_root = resolve_path(config.get("evaluation", {}).get("results_dir", "experiments/results"), root)
    summary = {mode: _summarize_mode(results_root, mode) for mode in modes}
    summary_path = results_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Ablation Summary", ""]
    for mode, item in summary.items():
        md.append(
            f"- {mode}: questions={item['question_count']}, valid={item['valid_question_count']}, errors={item['error_count']}, "
            f"citation_consistency={item['avg_citation_consistency_pct']}%, hallucination={item['avg_hallucination_pct']}%, "
            f"avg_quality_score={item['avg_quality_score']}"
        )
    if "no_rag" in summary and "rag_ts" in summary:
        no_rag = summary["no_rag"]
        rag_ts = summary["rag_ts"]
        md.extend(
            [
                "",
                "## Resume Metric",
                (
                    "相较 No-RAG 模式，RAG+TS-Feature 的引用一致性由 "
                    f"{no_rag['avg_citation_consistency_pct']}% 提升至 {rag_ts['avg_citation_consistency_pct']}%，"
                    f"幻觉率由 {no_rag['avg_hallucination_pct']}% 降至 {rag_ts['avg_hallucination_pct']}%。"
                ),
            ]
        )
    (results_root / "ablation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize existing TS-Explain ablation result files without API calls.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--modes", nargs="+", default=["no_rag", "rag_ts"])
    args = parser.parse_args()
    print(json.dumps(summarize_existing_results(config_path=args.config, modes=args.modes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
