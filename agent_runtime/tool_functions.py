from __future__ import annotations

from typing import Any, Dict, Optional

from agent_runtime.evidence import fact_card_to_item, item_to_retrieved_chunk, retrieved_chunk_to_item
from core.schema import ExplanationAnswer, ExplanationRequest
from evaluation.evaluator import Evaluator
from feature_extraction.real_window_resolver import RealWindowResolver
from knowledge_base.retrieval.query_rewriter import infer_filter_lists, infer_filters
from knowledge_base.retrieval.retriever import Retriever
from llm_interface.base import ExplanationService


def route_query(
    *,
    question: str,
    dataset_name: Optional[str],
    model_name: Optional[str],
    auto_window: bool,
) -> Dict[str, Any]:
    inferred = infer_filters(question)
    inferred_lists = infer_filter_lists(question)
    model_names = _merge_unique([model_name], inferred_lists["model_names"])
    dataset_names = _merge_unique([dataset_name], inferred_lists["dataset_names"])
    resolved_dataset = dataset_name or inferred.get("dataset_name")
    resolved_model = model_name or inferred.get("model_name")
    lower = question.lower()
    window_words = ["窗口", "window", "变量贡献", "贡献最大", "判为异常", "真实异常", "real window"]
    matched_window_words = [word for word in window_words if word in lower]
    needs_window = bool(auto_window or matched_window_words)
    question_type = "window_explanation" if needs_window else "dataset_model"
    if "比" in question or "compare" in lower:
        question_type = "comparison"
    mode = "rag-ts" if needs_window else "rag-only"

    uncertainty_reasons: list[str] = []
    confidence = 0.55
    if resolved_dataset and resolved_model:
        confidence += 0.25
    elif resolved_dataset or resolved_model:
        confidence += 0.10
        if not resolved_dataset:
            uncertainty_reasons.append("dataset_name is missing")
        if not resolved_model:
            uncertainty_reasons.append("model_name is missing")
    else:
        confidence -= 0.15
        uncertainty_reasons.append("dataset_name is missing")
        uncertainty_reasons.append("model_name is missing")

    if auto_window:
        confidence += 0.12
    elif matched_window_words:
        confidence += 0.10

    if question_type == "comparison":
        confidence += 0.05
        if len(model_names) >= 2:
            confidence += 0.12
        else:
            confidence -= 0.18
            uncertainty_reasons.append("comparison question has fewer than two models")
    elif question_type == "dataset_model":
        confidence -= 0.05
        uncertainty_reasons.append("question_type fell back to dataset_model")

    confidence = max(0.05, min(0.99, round(confidence, 3)))
    return {
        "route": {
            "question_type": question_type,
            "mode": mode,
            "needs_retrieval": True,
            "needs_window": needs_window,
            "dataset_name": resolved_dataset,
            "model_name": resolved_model,
            "model_names": model_names,
            "dataset_names": dataset_names,
            "rationale": "Heuristic routing based on query terms and explicit CLI flags.",
            "confidence": confidence,
            "router_type": "rule",
            "uncertainty_reasons": uncertainty_reasons,
        }
    }


def _merge_unique(explicit_values: list[Optional[str]], inferred_values: list[str]) -> list[str]:
    values: list[str] = []
    seen = set()
    for value in [*explicit_values, *inferred_values]:
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def retrieve_evidence(
    *,
    question: str,
    dataset_name: Optional[str],
    model_name: Optional[str],
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    hits = Retriever(config_path).retrieve(question, dataset_name=dataset_name, model_name=model_name)
    return {"retrieved": [retrieved_chunk_to_item(hit) for hit in hits]}


def analyze_real_window(
    *,
    dataset_name: Optional[str],
    model_name: Optional[str],
    window_length: int,
    window_strategy: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not dataset_name:
        raise ValueError("dataset_name is required for real window analysis")
    resolver = RealWindowResolver(config_path)
    real_window, card = resolver.extract_fact_card(
        dataset_name,
        model_name=model_name,
        window_length=window_length,
        strategy=window_strategy,  # type: ignore[arg-type]
    )
    evidence_item = fact_card_to_item(
        text=card.narrative,
        dataset_name=dataset_name,
        model_name=model_name,
        window_id=real_window.window_id,
    )
    return {
        "fact_card_text": card.narrative,
        "evidence_item": evidence_item,
        "window": {
            "dataset_name": real_window.dataset_name,
            "window_id": real_window.window_id,
            "start": real_window.start,
            "end": real_window.end,
            "label_density": real_window.label_density,
            "strategy": real_window.strategy,
            "source_path": real_window.source_path,
            "label_path": real_window.label_path,
        },
    }


def generate_answer(
    *,
    question: str,
    mode: str,
    dataset_name: Optional[str],
    model_name: Optional[str],
    provider: str,
    retrieved: list[Dict[str, Any]],
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    evidence = [item_to_retrieved_chunk(item) for item in retrieved]
    service = ExplanationService(config_path=config_path, provider_name=provider)
    answer = service.answer(
        ExplanationRequest(
            question=question,
            mode=mode,
            dataset_name=dataset_name,
            model_name=model_name,
            fact_card=None,
        ),
        retrieved_override=evidence,
    )
    return {
        "answer": answer.answer,
        "citations": answer.citations,
        "provider": answer.provider,
        "mode": answer.mode,
        "retrieved_count": len(answer.retrieved),
    }


def evaluate_answer(
    *,
    answer: str,
    citations: list[str],
    retrieved: list[Dict[str, Any]],
    mode: str,
    provider: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    evidence = [item_to_retrieved_chunk(item) for item in retrieved]
    answer_obj = ExplanationAnswer(
        answer=answer,
        citations=citations,
        retrieved=evidence,
        mode=mode,
        provider=provider,
    )
    return {"metrics": Evaluator(config_path).evaluate(answer_obj)}
