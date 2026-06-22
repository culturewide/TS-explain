from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent_runtime.schema import RouteDecision


LLM_ROUTER_DISABLED_REASON = "LLM router fallback is not enabled in offline mode."


def maybe_llm_route(
    *,
    question: str,
    current_route: RouteDecision,
    provider: str,
    config_path: str | Path | None = None,
) -> RouteDecision:
    """Return a hybrid routing decision with a safe placeholder LLM fallback.

    The deterministic rule router remains the main path. When confidence is high,
    this function returns the rule route unchanged. When confidence is low, the
    current implementation marks the route as hybrid and records why the optional
    LLM structured router did not run.

    Future implementation point:
    replace the low-confidence branch with a LangChain structured-output router
    or LangGraph conditional routing node that emits a RouteDecision-compatible
    object.
    """

    del question, provider, config_path
    if current_route.confidence >= 0.75:
        return current_route

    reasons = list(current_route.uncertainty_reasons)
    if LLM_ROUTER_DISABLED_REASON not in reasons:
        reasons.append(LLM_ROUTER_DISABLED_REASON)
    rationale = current_route.rationale
    if LLM_ROUTER_DISABLED_REASON not in rationale:
        rationale = f"{rationale} {LLM_ROUTER_DISABLED_REASON}".strip()
    return current_route.model_copy(
        update={
            "router_type": "hybrid",
            "rationale": rationale,
            "uncertainty_reasons": reasons,
        }
    )
