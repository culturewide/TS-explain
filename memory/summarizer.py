from __future__ import annotations

from typing import List, Set

from agent_runtime.schema import AgentState, MemorySummary


class MemoryCompressor:
    """Compress long step history into a compact run memory."""

    def summarize(self, state: AgentState) -> MemorySummary:
        evidence = self._key_evidence(state)
        decisions = self._decisions(state)
        failures = self._failures(state)

        summary = (
            f"Run {state.run_id} answered a question with provider={state.provider}. "
            f"Retrieved {len(state.retrieved)} evidence items, citations={state.citations}. "
            f"Final status={state.status}."
        )
        return MemorySummary(
            run_id=state.run_id,
            summary=summary,
            key_evidence=evidence,
            decisions=decisions,
            failures=failures,
        )

    def _key_evidence(self, state: AgentState) -> List[str]:
        evidence: List[str] = []
        seen: Set[str] = set()
        for item in state.retrieved:
            citation = item.get("citation_id") or item.get("id")
            source_type = item.get("source_type", "unknown")
            if citation and citation not in seen:
                evidence.append(f"{citation}:{source_type}")
                seen.add(citation)
        return evidence

    def _decisions(self, state: AgentState) -> List[str]:
        decisions: List[str] = []
        if state.route:
            decisions.append(
                f"route={state.route.question_type}, mode={state.route.mode}, "
                f"needs_retrieval={state.route.needs_retrieval}, needs_window={state.route.needs_window}, "
                f"confidence={state.route.confidence:.3f}, router_type={state.route.router_type}"
            )
            if state.route.uncertainty_reasons:
                decisions.append("uncertainty_reasons=" + "; ".join(state.route.uncertainty_reasons))
            if state.route.dataset_name or state.route.model_name:
                decisions.append(f"filters=dataset:{state.route.dataset_name}, model:{state.route.model_name}")
            if state.route.model_names or state.route.dataset_names:
                decisions.append(
                    "filter_lists="
                    f"models:{','.join(state.route.model_names) or 'none'}; "
                    f"datasets:{','.join(state.route.dataset_names) or 'none'}"
                )
            if state.route.question_type == "comparison":
                decisions.append(
                    "comparison_targets="
                    f"models:{','.join(state.route.model_names) or 'none'}; "
                    f"datasets:{','.join(state.route.dataset_names) or 'none'}"
                )

        has_s9001 = any(item.get("citation_id") == "S9001" for item in state.retrieved)
        decisions.append(f"real_window={bool(state.fact_card_text or has_s9001)}")
        decisions.append(f"provider_fallback={self._has_message(state, 'fallback')}")
        decisions.append(f"route_revised={self._has_message(state, 'revise route')}")

        for step in state.steps:
            if step.phase == "revise":
                decisions.append(f"revise:{step.message}")
        return decisions

    def _failures(self, state: AgentState) -> List[str]:
        failures = list(state.failures)
        for step in state.steps:
            obs = step.observation
            if obs and not obs.ok:
                failures.append(
                    f"agent={obs.agent}; tool={obs.tool_name}; "
                    f"error_type={obs.error_type}; error_message={obs.error_message}"
                )
        return failures

    def _has_message(self, state: AgentState, needle: str) -> bool:
        needle_lower = needle.lower()
        return any(needle_lower in step.message.lower() for step in state.steps)
