from __future__ import annotations

import re
from typing import List

from llm_interface.base import BaseLLMProvider, LLMMessage, LLMResponse


class OfflineProvider(BaseLLMProvider):
    provider_name = "offline"
    model_name = "retrieval-template"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        prompt = "\n\n".join(message.content for message in messages if message.role == "user")
        question_match = re.search(r"用户问题：\s*(.+)", prompt)
        question = question_match.group(1).strip() if question_match else "该问题"
        citations = re.findall(r"\[(S\d{4})\]\s*(.+)", prompt)
        fact_card_match = re.search(r"TS-Fact Card：(.+?)(?:\n\n|$)", prompt, flags=re.S)
        basis = "已检索证据和时序特征" if fact_card_match else "已检索证据"
        if not citations and not fact_card_match:
            basis = "模型参数化知识"
        lines = [f"结论：针对“{question}”，当前回答基于{basis}生成。"]
        if citations:
            metric_lines = self._metric_reasoning(citations)
            mechanism_lines = self._mechanism_reasoning(citations)
            lines.append("原因分析：")
            if metric_lines:
                lines.extend(metric_lines)
            if mechanism_lines:
                lines.extend(mechanism_lines)
            if not metric_lines and not mechanism_lines:
                lines.append("- 当前检索证据能提供相关记录，但离线模板无法形成更强因果解释；请使用 Qwen 或 DeepSeek 生成正式解释。")
            lines.append("关键依据：")
            for cid, text in citations[:6]:
                snippet = text.strip()
                if len(snippet) > 170:
                    snippet = snippet[:167] + "..."
                lines.append(f"- {snippet} [{cid}]")
        else:
            lines.append("依据：当前模式没有注入检索证据，因此只能给出方法性解释，不能声称具体实验事实。")
        if fact_card_match:
            fact = "TS-Fact Card：" + fact_card_match.group(1).strip()
            if len(fact) > 260:
                fact = fact[:257] + "..."
            lines.append(f"时序窗口解释：{fact}")
        lines.append("不确定性：若要用于论文或报告，请优先核对引用片段与原始实验表、模型 README 或数据集元数据是否一致。")
        return LLMResponse(content="\n".join(lines), provider=self.provider_name, model=self.model_name)

    def _metric_reasoning(self, citations: List[tuple[str, str]]) -> List[str]:
        lines: List[str] = []
        for cid, text in citations:
            metrics = []
            for key in ("F-score", "Affiliation_F1", "Precision", "Recall", "Accuracy", "AUC_ROC", "AUC_PR", "VUS_ROC", "VUS_PR"):
                match = re.search(rf"{re.escape(key)}=([0-9.]+)", text)
                if match:
                    metrics.append(f"{key}={match.group(1)}")
            if metrics:
                joined = "，".join(metrics[:5])
                lines.append(f"- 从实验结果看，相关指标为 {joined}，这能直接支撑“表现较好/较差”的判断 [{cid}]。")
                break
        return lines

    def _mechanism_reasoning(self, citations: List[tuple[str, str]]) -> List[str]:
        joined = "\n".join(f"[{cid}] {text}" for cid, text in citations)
        hints = [
            ("association discrepancy", "机制上，Anomaly Transformer 使用 association discrepancy 捕捉时间关联异常，因此高 Recall/F-score 可解释为它较容易捕获偏离正常时序依赖的片段"),
            ("channel fusion", "机制上，CATCH 的 channel fusion / channel-aware 设计会利用变量间关系，适合解释多变量相关结构变化"),
            ("frequency patching", "机制上，CATCH 的 frequency patching 会把异常解释转向频域片段和周期结构变化"),
            ("adaptive bottlenecks", "机制上，DADA 的 adaptive bottlenecks 和 dual adversarial decoders 适合解释跨域泛化场景"),
            ("dual attention", "机制上，DCdetector 的双注意力/对比思路会把异常解释落到多尺度表征差异上"),
            ("multiscale", "机制上，多尺度结构能解释趋势、周期或短时扰动在不同时间尺度上的异常贡献"),
            ("variates as tokens", "机制上，iTransformer 将变量作为 token，更适合讨论变量间依赖异常"),
        ]
        lines: List[str] = []
        for needle, reason in hints:
            for cid, text in citations:
                if needle in text.lower():
                    lines.append(f"- {reason} [{cid}]。")
                    return lines
        if not lines and any("paper_summary" in text or "architecture" in text for _, text in citations):
            lines.append("- 检索到了模型机制或架构摘要，但离线模板只能做轻量归纳；正式原因分析建议切换到 Qwen/DeepSeek。")
        return lines
