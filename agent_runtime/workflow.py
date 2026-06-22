from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from agent_runtime import tool_functions
from agent_runtime.router import maybe_llm_route
from agent_runtime.schema import AgentState, AgentStep, Observation, RouteDecision, RunRecord, ToolCall, utc_now
from agent_runtime.store import JsonlRunStore
from core.config import load_project_config, resolve_path
from memory.summarizer import MemoryCompressor
from sandbox.tool_sandbox import ToolSandbox


class BaseAgent:
    name = "BaseAgent"

    def __init__(self, workflow: "MultiAgentWorkflow"):
        self.workflow = workflow

    def tool_call(self, tool_name: str, args: Dict[str, Any], timeout_seconds: float = 60.0) -> ToolCall:
        return ToolCall(
            agent=self.name,  # type: ignore[arg-type]
            tool_name=tool_name,
            args=args,
            timeout_seconds=timeout_seconds,
            cwd=str(self.workflow.project_root),
        )


class QueryRouter(BaseAgent):
    name = "QueryRouter"

    def run(self, state: AgentState) -> Observation:
        call = self.tool_call(
            "route_query",
            {
                "question": state.question,
                "dataset_name": state.dataset_name,
                "model_name": state.model_name,
                "auto_window": state.auto_window,
            },
            timeout_seconds=10,
        )
        return self.workflow.execute_tool(state, call, "Route question into workflow needs")


class EvidenceRetriever(BaseAgent):
    name = "EvidenceRetriever"

    def run(self, state: AgentState) -> Observation:
        if state.route and state.route.question_type == "comparison" and len(state.route.model_names) >= 2:
            return self._run_comparison_retrieval(state)
        call = self.tool_call(
            "retrieve_evidence",
            {
                "question": state.question,
                "dataset_name": state.route.dataset_name if state.route else state.dataset_name,
                "model_name": state.route.model_name if state.route else state.model_name,
                "config_path": self.workflow.config_path,
            },
            timeout_seconds=60,
        )
        return self.workflow.execute_tool(state, call, "Retrieve RAG evidence")

    def _run_comparison_retrieval(self, state: AgentState) -> Observation:
        assert state.route is not None
        merged: list[Dict[str, Any]] = []
        seen = set()
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        dataset_targets = state.route.dataset_names or [state.route.dataset_name]
        for dataset_name in dataset_targets:
            for model_name in state.route.model_names:
                call = self.tool_call(
                    "retrieve_evidence",
                    {
                        "question": state.question,
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "config_path": self.workflow.config_path,
                    },
                    timeout_seconds=60,
                )
                obs = self.workflow.execute_tool(
                    state,
                    call,
                    f"Retrieve comparison RAG evidence for model={model_name}, dataset={dataset_name}",
                )
                stdout_parts.append(obs.stdout)
                stderr_parts.append(obs.stderr)
                if not obs.ok:
                    return obs
                for item in obs.output.get("retrieved", []):
                    key = item.get("citation_id") or item.get("id")
                    if key and key not in seen:
                        seen.add(key)
                        merged.append(item)
        return Observation(
            tool_call_id="comparison-merged-retrieval",
            agent=self.name,
            tool_name="retrieve_evidence",
            ok=True,
            output={"retrieved": merged},
            stdout="\n".join(part for part in stdout_parts if part),
            stderr="\n".join(part for part in stderr_parts if part),
        )


class WindowAnalyzer(BaseAgent):
    name = "WindowAnalyzer"

    def run(self, state: AgentState) -> Observation:
        dataset_name = state.route.dataset_name if state.route else state.dataset_name
        call = self.tool_call(
            "analyze_real_window",
            {
                "dataset_name": dataset_name,
                "model_name": state.route.model_name if state.route else state.model_name,
                "window_length": state.window_length,
                "window_strategy": state.window_strategy,
                "config_path": self.workflow.config_path,
            },
            timeout_seconds=90,
        )
        return self.workflow.execute_tool(state, call, "Analyze real dataset window")


class AnswerWriter(BaseAgent):
    name = "AnswerWriter"

    def run(self, state: AgentState) -> Observation:
        mode = state.route.mode if state.route else ("rag-ts" if state.fact_card_text else "rag-only")
        call = self.tool_call(
            "generate_answer",
            {
                "question": state.question,
                "mode": mode,
                "dataset_name": state.route.dataset_name if state.route else state.dataset_name,
                "model_name": state.route.model_name if state.route else state.model_name,
                "provider": state.provider,
                "retrieved": state.retrieved,
                "config_path": self.workflow.config_path,
            },
            timeout_seconds=120,
        )
        return self.workflow.execute_tool(state, call, "Generate cited answer")


class CitationAuditor(BaseAgent):
    name = "CitationAuditor"

    def run(self, state: AgentState) -> Observation:
        mode = state.route.mode if state.route else ("rag-ts" if state.fact_card_text else "rag-only")
        call = self.tool_call(
            "evaluate_answer",
            {
                "answer": state.answer or "",
                "citations": state.citations,
                "retrieved": state.retrieved,
                "mode": mode,
                "provider": state.provider,
                "config_path": self.workflow.config_path,
            },
            timeout_seconds=30,
        )
        return self.workflow.execute_tool(state, call, "Evaluate answer and citations")


class Supervisor(BaseAgent):
    name = "Supervisor"

    def run(self, state: AgentState) -> AgentState:
        state.status = "running"
        self.workflow.persist_state(state, "run_started")

        route_obs = self.workflow.query_router.run(state)
        if not route_obs.ok:
            return self.workflow.fail(state, route_obs)
        rule_route = RouteDecision.model_validate(route_obs.output["route"])
        state.route = maybe_llm_route(
            question=state.question,
            current_route=rule_route,
            provider=state.provider,
            config_path=self.workflow.config_path,
        )
        state.dataset_name = state.route.dataset_name
        state.model_name = state.route.model_name

        if state.route.needs_retrieval:
            evidence_obs = self.workflow.evidence_retriever.run(state)
            if not evidence_obs.ok:
                return self.workflow.fail(state, evidence_obs)
            state.retrieved = evidence_obs.output.get("retrieved", [])

        if state.route.needs_window:
            window_obs = self.workflow.window_analyzer.run(state)
            if not window_obs.ok:
                state.failures.append(f"WindowAnalyzer failed: {window_obs.error_type}: {window_obs.error_message}")
                revise = AgentStep(
                    index=len(state.steps) + 1,
                    agent="Supervisor",
                    phase="revise",
                    status="succeeded",
                    message="Real window failed; revise route to rag-only.",
                )
                state.steps.append(revise)
                state.route.needs_window = False
                state.route.mode = "rag-only"
                self.workflow.persist_state(state, "route_revised")
            else:
                state.fact_card_text = window_obs.output.get("fact_card_text")
                evidence_item = window_obs.output.get("evidence_item")
                if evidence_item:
                    state.retrieved = [
                        item for item in state.retrieved if item.get("citation_id") != evidence_item.get("citation_id")
                    ]
                    state.retrieved.insert(0, evidence_item)
                self.workflow.persist_state(state, "window_evidence_attached")

        answer_obs = self.workflow.answer_writer.run(state)
        if not answer_obs.ok and state.provider != "offline":
            state.failures.append(f"AnswerWriter provider failed: {answer_obs.error_type}: {answer_obs.error_message}")
            state.provider = "offline"
            fallback_step = AgentStep(
                index=len(state.steps) + 1,
                agent="Supervisor",
                phase="revise",
                status="succeeded",
                message="Provider failed; fallback to offline provider.",
            )
            state.steps.append(fallback_step)
            self.workflow.persist_state(state, "provider_fallback")
            answer_obs = self.workflow.answer_writer.run(state)
        if not answer_obs.ok:
            return self.workflow.fail(state, answer_obs)
        state.answer = answer_obs.output.get("answer", "")
        state.citations = answer_obs.output.get("citations", [])

        audit_obs = self.workflow.citation_auditor.run(state)
        if not audit_obs.ok:
            return self.workflow.fail(state, audit_obs)
        state.metrics = audit_obs.output.get("metrics", {})

        state.status = "succeeded"
        state.updated_at = utc_now()
        state.memory_summary = self.workflow.memory.summarize(state)
        finish_step = AgentStep(
            index=len(state.steps) + 1,
            agent="Supervisor",
            phase="finish",
            status="succeeded",
            message="Workflow completed.",
        )
        state.steps.append(finish_step)
        self.workflow.persist_state(state, "run_finished")
        self.workflow.persist_final(state)
        return state


class MultiAgentWorkflow:
    """Plan-action-observe-evaluate workflow built on existing TS-Explain modules."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        provider: str = "offline",
        runs_dir: str | Path | None = None,
    ):
        self.config = load_project_config(config_path)
        self.project_root = Path(self.config["_project_root"])
        self.config_path = self.config["_config_path"]
        self.provider = provider
        default_runs = resolve_path("experiments/agent_runs", self.project_root)
        self.store = JsonlRunStore(runs_dir or default_runs)
        self.memory = MemoryCompressor()
        self.sandbox = ToolSandbox(allowed_root=self.project_root)
        self._register_tools()

        self.query_router = QueryRouter(self)
        self.evidence_retriever = EvidenceRetriever(self)
        self.window_analyzer = WindowAnalyzer(self)
        self.answer_writer = AnswerWriter(self)
        self.citation_auditor = CitationAuditor(self)
        self.supervisor = Supervisor(self)

    def run(
        self,
        *,
        question: str,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        auto_window: bool = False,
        window_length: int = 256,
        window_strategy: str = "max_label_density",
    ) -> AgentState:
        state = AgentState(
            question=question,
            provider=self.provider,
            dataset_name=dataset_name,
            model_name=model_name,
            auto_window=auto_window,
            window_length=window_length,
            window_strategy=window_strategy,
        )
        return self.supervisor.run(state)

    def execute_tool(self, state: AgentState, call: ToolCall, message: str) -> Observation:
        step = AgentStep(
            index=len(state.steps) + 1,
            agent=call.agent,
            phase="tool",
            status="running",
            message=message,
            tool_call=call,
        )
        state.steps.append(step)
        self.persist_state(state, "tool_started")
        obs = self.sandbox.run(call)
        step.status = "succeeded" if obs.ok else "failed"
        step.phase = "observe"
        step.observation = obs
        step.message = message if obs.ok else f"{message} failed: {obs.error_message}"
        state.updated_at = utc_now()
        self.persist_state(state, "tool_finished")
        return obs

    def persist_state(self, state: AgentState, event_type: str) -> None:
        self.store.save_state(state)
        self.store.append_event(
            state.run_id,
            event_type,
            {"status": state.status, "step_count": len(state.steps), "state": state.model_dump(mode="json")},
        )

    def persist_final(self, state: AgentState) -> None:
        record = RunRecord(
            run_id=state.run_id,
            question=state.question,
            provider=state.provider,
            status=state.status,
            output_dir=str(self.store.run_dir(state.run_id)),
            final_answer=state.answer,
            metrics=state.metrics,
            memory_summary=state.memory_summary,
            created_at=state.created_at,
            updated_at=state.updated_at,
        )
        self.store.save_final(record)

    def fail(self, state: AgentState, obs: Observation) -> AgentState:
        state.status = "failed"
        state.failures.append(f"{obs.agent}.{obs.tool_name}: {obs.error_type}: {obs.error_message}")
        state.updated_at = utc_now()
        state.memory_summary = self.memory.summarize(state)
        self.persist_state(state, "run_failed")
        self.persist_final(state)
        return state

    def _register_tools(self) -> None:
        self.sandbox.register("route_query", tool_functions.route_query)
        self.sandbox.register("retrieve_evidence", tool_functions.retrieve_evidence)
        self.sandbox.register("analyze_real_window", tool_functions.analyze_real_window)
        self.sandbox.register("generate_answer", tool_functions.generate_answer)
        self.sandbox.register("evaluate_answer", tool_functions.evaluate_answer)
