from __future__ import annotations

from typing import Iterable, List

from core.schema import RetrievedChunk
from llm_interface.prompts.system_prompts import SYSTEM_PROMPT_ZH
from llm_interface.types import LLMMessage


def retrieved_context(chunks: Iterable[RetrievedChunk]) -> str:
    lines = []
    for hit in chunks:
        cid = hit.citation_id
        meta = hit.chunk.metadata
        source = f"{hit.chunk.source_type}:{hit.chunk.source_path}"
        model = meta.get("model_name") or "未知模型"
        dataset = meta.get("dataset_name") or "未知数据集"
        lines.append(f"[{cid}] {hit.chunk.text}\n来源={source}; 模型={model}; 数据集={dataset}; score={hit.score:.3f}")
    return "\n\n".join(lines)


def build_explanation_messages(
    *,
    question: str,
    mode: str,
    retrieval_context: str = "",
    fact_card_text: str = "",
) -> List[LLMMessage]:
    user = [
        f"运行模式：{mode}",
        f"用户问题：{question}",
        "回答要求：",
        "1. 使用中文回答。",
        "2. 事实性陈述必须带引用编号；若没有证据，标注证据不足。",
        "3. 区分实验结果、模型机制、数据集属性和时序窗口特征。",
        "4. 不要把 TS-Fact Card 当作训练集全局事实，它只描述当前窗口。",
    ]
    if retrieval_context:
        user.append("检索证据：\n" + retrieval_context)
    else:
        user.append("检索证据：无。")
    if fact_card_text:
        user.append("TS-Fact Card：\n" + fact_card_text)
    return [LLMMessage(role="system", content=SYSTEM_PROMPT_ZH), LLMMessage(role="user", content="\n\n".join(user))]
