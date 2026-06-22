from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import load_project_config, load_yaml, resolve_path
from core.schema import ExplanationRequest, QuestionItem
from evaluation.evaluator import Evaluator
from feature_extraction.extractor import TSFeatureExtractor
from llm_interface.base import ExplanationService


MODE_MAP = {
    "no_rag": "no-rag",
    "rag_only": "rag-only",
    "rag_ts": "rag-ts",
}


def load_question_bank(path: str | Path) -> List[QuestionItem]:
    data = load_yaml(path)
    return [QuestionItem(**item) for item in data.get("questions", [])]


def synthetic_fact_card(question: QuestionItem, seed: int = 42):
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in question.id))
    length = 96
    t = np.arange(length)
    base = np.stack(
        [
            0.02 * t + np.sin(2 * np.pi * t / 24),
            np.cos(2 * np.pi * t / 16),
            rng.normal(0, 0.2, length),
        ],
        axis=1,
    )
    base[55:63, 1] += 3.0
    scores = np.mean(np.abs(base - base.mean(axis=0)), axis=1)
    extractor = TSFeatureExtractor()
    return extractor.extract(base, window_id=f"synthetic-{question.id}", anomaly_scores=scores, feature_names=["load", "pressure", "noise"])


def run_ablation(
    *,
    config_path: str | Path | None = None,
    provider_name: str = "offline",
    limit: Optional[int] = None,
    modes: Optional[Sequence[str]] = None,
) -> dict:
    config = load_project_config(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    root = Path(config["_project_root"])
    question_path = resolve_path(config.get("evaluation", {}).get("question_bank", "experiments/question_bank.yaml"), root)
    questions = load_question_bank(question_path)
    if limit:
        questions = questions[:limit]

    service = ExplanationService(config_path, provider_name=provider_name)
    evaluator = Evaluator(config_path)
    results_root = resolve_path(config.get("evaluation", {}).get("results_dir", "experiments/results"), root)
    results_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    selected_modes = list(modes) if modes else list(MODE_MAP.keys())
    invalid_modes = [mode for mode in selected_modes if mode not in MODE_MAP]
    if invalid_modes:
        raise ValueError(f"Unknown ablation mode(s): {', '.join(invalid_modes)}")

    for output_dir in selected_modes:
        mode = MODE_MAP[output_dir]
        mode_dir = results_root / output_dir
        mode_dir.mkdir(parents=True, exist_ok=True)
        out_path = mode_dir / "answers.jsonl"
        progress_path = mode_dir / "progress.jsonl"
        records = []
        with out_path.open("w", encoding="utf-8") as out_f, progress_path.open("w", encoding="utf-8") as progress_f:
            for idx, question in enumerate(questions, start=1):
                started = time.time()
                print(f"[{output_dir}] {idx}/{len(questions)} {question.id} start", flush=True)
                progress_f.write(
                    json.dumps(
                        {
                            "mode": output_dir,
                            "index": idx,
                            "total": len(questions),
                            "id": question.id,
                            "event": "start",
                            "time": round(started, 3),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                progress_f.flush()
                try:
                    fact_card = synthetic_fact_card(question, seed) if mode == "rag-ts" else None
                    answer = service.answer(
                        ExplanationRequest(
                            question=question.question,
                            mode=mode,
                            dataset_name=question.dataset_name,
                            model_name=question.model_name,
                            fact_card=fact_card,
                        )
                    )
                    metrics = evaluator.evaluate(answer)
                    elapsed = round(time.time() - started, 3)
                    record = {
                        "id": question.id,
                        "category": question.category,
                        "question": question.question,
                        "answer": answer.answer,
                        "citations": answer.citations,
                        "metrics": metrics,
                        "elapsed_seconds": elapsed,
                    }
                    records.append(record)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    progress_f.write(
                        json.dumps(
                            {
                                "mode": output_dir,
                                "index": idx,
                                "total": len(questions),
                                "id": question.id,
                                "event": "done",
                                "elapsed_seconds": elapsed,
                                "time": round(time.time(), 3),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    progress_f.flush()
                    print(f"[{output_dir}] {idx}/{len(questions)} {question.id} done in {elapsed}s", flush=True)
                except Exception as exc:
                    elapsed = round(time.time() - started, 3)
                    record = {
                        "id": question.id,
                        "category": question.category,
                        "question": question.question,
                        "answer": "",
                        "citations": [],
                        "metrics": {},
                        "elapsed_seconds": elapsed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    records.append(record)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    progress_f.write(
                        json.dumps(
                            {
                                "mode": output_dir,
                                "index": idx,
                                "total": len(questions),
                                "id": question.id,
                                "event": "error",
                                "elapsed_seconds": elapsed,
                                "error": f"{type(exc).__name__}: {exc}",
                                "time": round(time.time(), 3),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    progress_f.flush()
                    print(f"[{output_dir}] {idx}/{len(questions)} {question.id} error after {elapsed}s: {exc}", flush=True)
        valid_records = [r for r in records if r.get("metrics") and "hallucination" in r["metrics"]]
        avg_h = float(np.mean([r["metrics"]["hallucination"]["hallucination_rate"] for r in valid_records])) if valid_records else 0.0
        avg_q = float(np.mean([r["metrics"]["answer_quality"]["score"] for r in valid_records])) if valid_records else 0.0
        avg_c = float(np.mean([r["metrics"]["citation_consistency_rate"] for r in valid_records])) if valid_records else 0.0
        summary[output_dir] = {
            "question_count": len(records),
            "valid_question_count": len(valid_records),
            "error_count": len(records) - len(valid_records),
            "avg_hallucination_rate": round(avg_h, 4),
            "avg_hallucination_pct": round(avg_h * 100, 2),
            "avg_citation_consistency_rate": round(avg_c, 4),
            "avg_citation_consistency_pct": round(avg_c * 100, 2),
            "avg_quality_score": round(avg_q, 4),
            "path": str(out_path),
        }
    summary_path = results_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Ablation Summary", ""]
    for mode, item in summary.items():
        md.append(
            f"- {mode}: questions={item['question_count']}, citation_consistency={item['avg_citation_consistency_pct']}%, hallucination={item['avg_hallucination_pct']}%, avg_quality_score={item['avg_quality_score']}"
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
    parser = argparse.ArgumentParser(description="Run TS-Explain ablation modes.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--provider", default="offline", choices=["offline", "qwen", "deepseek"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        choices=list(MODE_MAP.keys()),
        help="Ablation modes to run, e.g. --modes no_rag rag_ts",
    )
    args = parser.parse_args()
    summary = run_ablation(config_path=args.config, provider_name=args.provider, limit=args.limit, modes=args.modes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
