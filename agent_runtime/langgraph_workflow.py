from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

from pydantic import BaseModel, Field

from agent_runtime import tool_functions
from agent_runtime.router import maybe_llm_route
from agent_runtime.schema import AgentState, AgentStep, Observation, RouteDecision, RunRecord, ToolCall, utc_now
from agent_runtime.store import JsonlRunStore
from core.config import load_project_config, resolve_path
from memory.summarizer import MemoryCompressor
from sandbox.tool_sandbox import ToolSandbox

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
except ModuleNotFoundError:  # pragma: no cover - exercised only when dependency is missing
    END = START = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    InMemorySaver = None  # type: ignore[assignment]
    Command = None  # type: ignore[assignment]
    interrupt = None  # type: ignore[assignment]


class GraphState(TypedDict, total=False):
    state: Dict[str, Any]
    options: Dict[str, Any]


class LangGraphRunResult(BaseModel):
    state: AgentState
    thread_id: str
    interrupted: bool = False
    interrupts: list[Dict[str, Any]] = Field(default_factory=list)
    output_dir: str


class LangGraphWorkflow:
    """LangGraph StateGraph version of the TS-Explain agent workflow.

    This class intentionally runs beside, not instead of, MultiAgentWorkflow.
    It reuses the same Pydantic state, sandboxed tools, memory compressor, and
    JSONL store while expressing branch control through LangGraph edges.
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        provider: str = "offline",
        runs_dir: str | Path | None = None,
        enable_checkpoint: bool = True,
        checkpoint_backend: str = "memory",
        checkpoint_path: str | Path | None = None,
    ):
        if StateGraph is None:
            raise RuntimeError("langgraph is not installed. Install requirements.txt or `pip install langgraph>=0.2`.")
        self.config = load_project_config(config_path)
        self.project_root = Path(self.config["_project_root"])
        self.config_path = self.config["_config_path"]
        self.provider = provider
        default_runs = resolve_path("experiments/agent_runs", self.project_root)
        self.store = JsonlRunStore(runs_dir or default_runs)
        self.memory = MemoryCompressor()
        self.sandbox = ToolSandbox(allowed_root=self.project_root)
        self.enable_checkpoint = enable_checkpoint
        self.requested_checkpoint_backend = checkpoint_backend
        self.checkpoint_backend = checkpoint_backend
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.checkpoint_notice = ""
        self._checkpoint_context = None
        self.checkpointer = self._build_checkpointer()
        self._register_tools()
        self.graph = self._build_graph()

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
        result = self.run_interruptible(
            question=question,
            dataset_name=dataset_name,
            model_name=model_name,
            auto_window=auto_window,
            window_length=window_length,
            window_strategy=window_strategy,
            human_review=False,
        )
        return result.state

    def run_interruptible(
        self,
        *,
        question: str,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        auto_window: bool = False,
        window_length: int = 256,
        window_strategy: str = "max_label_density",
        human_review: bool = True,
        thread_id: Optional[str] = None,
    ) -> LangGraphRunResult:
        state = AgentState(
            question=question,
            provider=self.provider,
            dataset_name=dataset_name,
            model_name=model_name,
            auto_window=auto_window,
            window_length=window_length,
            window_strategy=window_strategy,
        )
        resolved_thread_id = thread_id or state.run_id
        config = self._thread_config(resolved_thread_id)
        self._record_checkpoint_event(state, resolved_thread_id)
        result = self.graph.invoke(
            {"state": state.model_dump(mode="json"), "options": {"human_review": human_review}},
            config=config,
        )
        run_result = self._to_run_result(result, resolved_thread_id, config)
        if run_result.interrupted:
            self.persist_state(run_result.state, "langgraph_interrupted")
        return run_result

    def resume_interrupt(self, thread_id: str, resume: dict[str, Any] | bool | str) -> LangGraphRunResult:
        if Command is None:
            raise RuntimeError("langgraph Command is not available.")
        config = self._thread_config(thread_id)
        snapshot = self.graph.get_state(config)
        if not snapshot.values or "state" not in snapshot.values:
            raise ValueError(f"No checkpointed state found for thread_id={thread_id}")
        state = AgentState.model_validate(snapshot.values["state"])
        self.store.append_event(
            state.run_id,
            "langgraph_resume_started",
            {"status": state.status, "thread_id": thread_id, "resume": resume},
        )
        self.store.append_event(
            state.run_id,
            "langgraph_review_resumed",
            {"status": state.status, "thread_id": thread_id, "resume": resume},
        )
        result = self.graph.invoke(Command(resume=resume), config=config)
        run_result = self._to_run_result(result, thread_id, config)
        if run_result.interrupted:
            self.persist_state(run_result.state, "langgraph_interrupted")
        self.store.append_event(
            run_result.state.run_id,
            "langgraph_resume_finished",
            {
                "status": run_result.state.status,
                "thread_id": thread_id,
                "interrupted": run_result.interrupted,
                "state": run_result.state.model_dump(mode="json"),
            },
        )
        return run_result

    def route_node(self, graph_state: GraphState) -> GraphState:
        state = self._load_state(graph_state)
        state.status = "running"
        self.persist_state(state, "langgraph_route_started")
        call = self.tool_call(
            "QueryRouter",
            "route_query",
            {
                "question": state.question,
                "dataset_name": state.dataset_name,
                "model_name": state.model_name,
                "auto_window": state.auto_window,
            },
            timeout_seconds=10,
        )
        obs = self.execute_tool(state, call, "Route question into workflow needs")
        if not obs.ok:
            state = self.mark_failed(state, obs)
            self.persist_state(state, "langgraph_route_finished")
            return self._dump_state(state, graph_state)

        rule_route = RouteDecision.model_validate(obs.output["route"])
        state.route = maybe_llm_route(
            question=state.question,
            current_route=rule_route,
            provider=state.provider,
            config_path=self.config_path,
        )
        state.dataset_name = state.route.dataset_name
        state.model_name = state.route.model_name
        self.persist_state(state, "langgraph_route_finished")
        return self._dump_state(state, graph_state)

    def retrieve_node(self, graph_state: GraphState) -> GraphState:
        state = self._load_state(graph_state)
        self.persist_state(state, "langgraph_retrieve_started")
        if state.route and state.route.question_type == "comparison" and len(state.route.model_names) >= 2:
            obs = self._run_comparison_retrieval(state)
        else:
            call = self.tool_call(
                "EvidenceRetriever",
                "retrieve_evidence",
                {
                    "question": state.question,
                    "dataset_name": state.route.dataset_name if state.route else state.dataset_name,
                    "model_name": state.route.model_name if state.route else state.model_name,
                    "config_path": self.config_path,
                },
                timeout_seconds=60,
            )
            obs = self.execute_tool(state, call, "Retrieve RAG evidence")

        if not obs.ok:
            state = self.mark_failed(state, obs)
        else:
            state.retrieved = obs.output.get("retrieved", [])
        self.persist_state(state, "langgraph_retrieve_finished")
        return self._dump_state(state, graph_state)

    def window_node(self, graph_state: GraphState) -> GraphState:
        state = self._load_state(graph_state)
        self.persist_state(state, "langgraph_window_started")
        call = self.tool_call(
            "WindowAnalyzer",
            "analyze_real_window",
            {
                "dataset_name": state.route.dataset_name if state.route else state.dataset_name,
                "model_name": state.route.model_name if state.route else state.model_name,
                "window_length": state.window_length,
                "window_strategy": state.window_strategy,
                "config_path": self.config_path,
            },
            timeout_seconds=90,
        )
        obs = self.execute_tool(state, call, "Analyze real dataset window")
        if not obs.ok:
            state.failures.append(f"WindowAnalyzer failed: {obs.error_type}: {obs.error_message}")
            if state.route:
                state.route.needs_window = False
                state.route.mode = "rag-only"
            state.steps.append(
                AgentStep(
                    index=len(state.steps) + 1,
                    agent="Supervisor",
                    phase="revise",
                    status="succeeded",
                    message="Real window failed; revise route to rag-only.",
                )
            )
            self.persist_state(state, "langgraph_route_revised")
        else:
            state.fact_card_text = obs.output.get("fact_card_text")
            evidence_item = obs.output.get("evidence_item")
            if evidence_item:
                state.retrieved = [
                    item for item in state.retrieved if item.get("citation_id") != evidence_item.get("citation_id")
                ]
                state.retrieved.insert(0, evidence_item)
        self.persist_state(state, "langgraph_window_finished")
        return self._dump_state(state, graph_state)

    def review_node(self, graph_state: GraphState) -> GraphState:
        state = self._load_state(graph_state)
        if not self._options(graph_state).get("human_review", False):
            return self._dump_state(state, graph_state)
        self.persist_state(state, "langgraph_review_started")
        payload = self._review_payload(state)
        decision = interrupt(payload)
        action = self._normalize_review_decision(decision)
        if action["action"] == "approve":
            pass
        elif action["action"] == "drop_citations":
            drop_ids = set(action.get("drop_citation_ids", []))
            state.retrieved = [
                item for item in state.retrieved if (item.get("citation_id") or item.get("id")) not in drop_ids
            ]
        elif action["action"] == "edit_question":
            if action.get("question"):
                state.question = str(action["question"])
        elif action["action"] == "cancel":
            state.status = "failed"
            reason = action.get("reason") or "Human review cancelled before answer generation."
            state.failures.append(f"Human review cancelled: {reason}")
        else:
            state.status = "failed"
            state.failures.append(f"Unknown human review action: {action['action']}")
        self.persist_state(state, "langgraph_review_finished")
        return self._dump_state(state, graph_state)

    def answer_node(self, graph_state: GraphState) -> GraphState:
        state = self._load_state(graph_state)
        self.persist_state(state, "langgraph_answer_started")
        obs = self._run_answer_tool(state)
        if not obs.ok and state.provider != "offline":
            state.failures.append(f"AnswerWriter provider failed: {obs.error_type}: {obs.error_message}")
            state.provider = "offline"
            state.steps.append(
                AgentStep(
                    index=len(state.steps) + 1,
                    agent="Supervisor",
                    phase="revise",
                    status="succeeded",
                    message="Provider failed; fallback to offline provider.",
                )
            )
            self.persist_state(state, "langgraph_provider_fallback")
            obs = self._run_answer_tool(state)

        if not obs.ok:
            state = self.mark_failed(state, obs)
        else:
            state.answer = obs.output.get("answer", "")
            state.citations = obs.output.get("citations", [])
        self.persist_state(state, "langgraph_answer_finished")
        return self._dump_state(state, graph_state)

    def audit_node(self, graph_state: GraphState) -> GraphState:
        state = self._load_state(graph_state)
        self.persist_state(state, "langgraph_audit_started")
        mode = state.route.mode if state.route else ("rag-ts" if state.fact_card_text else "rag-only")
        call = self.tool_call(
            "CitationAuditor",
            "evaluate_answer",
            {
                "answer": state.answer or "",
                "citations": state.citations,
                "retrieved": state.retrieved,
                "mode": mode,
                "provider": state.provider,
                "config_path": self.config_path,
            },
            timeout_seconds=30,
        )
        obs = self.execute_tool(state, call, "Evaluate answer and citations")
        if not obs.ok:
            state = self.mark_failed(state, obs)
        else:
            state.metrics = obs.output.get("metrics", {})
        self.persist_state(state, "langgraph_audit_finished")
        return self._dump_state(state, graph_state)

    def finish_node(self, graph_state: GraphState) -> GraphState:
        state = self._load_state(graph_state)
        self.persist_state(state, "langgraph_finish_started")
        state.status = "succeeded"
        state.updated_at = utc_now()
        state.memory_summary = self.memory.summarize(state)
        state.steps.append(
            AgentStep(
                index=len(state.steps) + 1,
                agent="Supervisor",
                phase="finish",
                status="succeeded",
                message="LangGraph workflow completed.",
            )
        )
        self.persist_state(state, "langgraph_run_finished")
        self.persist_final(state)
        return self._dump_state(state, graph_state)

    def fail_node(self, graph_state: GraphState) -> GraphState:
        state = self._load_state(graph_state)
        self.persist_state(state, "langgraph_fail_started")
        state.status = "failed"
        state.updated_at = utc_now()
        state.memory_summary = self.memory.summarize(state)
        self.persist_state(state, "langgraph_run_failed")
        self.persist_final(state)
        return self._dump_state(state, graph_state)

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

    def mark_failed(self, state: AgentState, obs: Observation) -> AgentState:
        state.status = "failed"
        state.failures.append(f"{obs.agent}.{obs.tool_name}: {obs.error_type}: {obs.error_message}")
        state.updated_at = utc_now()
        return state

    def tool_call(
        self,
        agent: str,
        tool_name: str,
        args: Dict[str, Any],
        timeout_seconds: float = 60.0,
    ) -> ToolCall:
        return ToolCall(
            agent=agent,  # type: ignore[arg-type]
            tool_name=tool_name,
            args=args,
            timeout_seconds=timeout_seconds,
            cwd=str(self.project_root),
        )

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

    def get_state(self, thread_id: str):
        return self.graph.get_state(self._thread_config(thread_id))

    def get_state_history(self, thread_id: str):
        return list(self.graph.get_state_history(self._thread_config(thread_id)))

    def close(self) -> None:
        if self._checkpoint_context is not None:
            self._checkpoint_context.__exit__(None, None, None)
            self._checkpoint_context = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _run_answer_tool(self, state: AgentState) -> Observation:
        mode = state.route.mode if state.route else ("rag-ts" if state.fact_card_text else "rag-only")
        call = self.tool_call(
            "AnswerWriter",
            "generate_answer",
            {
                "question": state.question,
                "mode": mode,
                "dataset_name": state.route.dataset_name if state.route else state.dataset_name,
                "model_name": state.route.model_name if state.route else state.model_name,
                "provider": state.provider,
                "retrieved": state.retrieved,
                "config_path": self.config_path,
            },
            timeout_seconds=120,
        )
        return self.execute_tool(state, call, "Generate cited answer")

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
                    "EvidenceRetriever",
                    "retrieve_evidence",
                    {
                        "question": state.question,
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "config_path": self.config_path,
                    },
                    timeout_seconds=60,
                )
                obs = self.execute_tool(
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
            agent="EvidenceRetriever",
            tool_name="retrieve_evidence",
            ok=True,
            output={"retrieved": merged},
            stdout="\n".join(part for part in stdout_parts if part),
            stderr="\n".join(part for part in stderr_parts if part),
        )

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("route", self.route_node)
        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("window", self.window_node)
        builder.add_node("review", self.review_node)
        builder.add_node("answer", self.answer_node)
        builder.add_node("audit", self.audit_node)
        builder.add_node("finish", self.finish_node)
        builder.add_node("fail", self.fail_node)

        builder.add_edge(START, "route")
        builder.add_conditional_edges(
            "route",
            self._after_route,
            {"retrieve": "retrieve", "review": "review", "answer": "answer", "fail": "fail"},
        )
        builder.add_conditional_edges(
            "retrieve",
            self._after_retrieve,
            {"window": "window", "review": "review", "answer": "answer", "fail": "fail"},
        )
        builder.add_conditional_edges(
            "window",
            self._after_window,
            {"review": "review", "answer": "answer", "fail": "fail"},
        )
        builder.add_conditional_edges("review", self._after_review, {"answer": "answer", "fail": "fail"})
        builder.add_conditional_edges("answer", self._after_answer, {"audit": "audit", "fail": "fail"})
        builder.add_conditional_edges("audit", self._after_audit, {"finish": "finish", "fail": "fail"})
        builder.add_edge("finish", END)
        builder.add_edge("fail", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _after_route(self, graph_state: GraphState) -> str:
        state = self._load_state(graph_state)
        if state.status == "failed":
            return "fail"
        if state.route and state.route.needs_retrieval:
            return "retrieve"
        return "review" if self._should_review(graph_state) else "answer"

    def _after_retrieve(self, graph_state: GraphState) -> str:
        state = self._load_state(graph_state)
        if state.status == "failed":
            return "fail"
        if state.route and state.route.needs_window:
            return "window"
        return "review" if self._should_review(graph_state) else "answer"

    def _after_window(self, graph_state: GraphState) -> str:
        state = self._load_state(graph_state)
        if state.status == "failed":
            return "fail"
        return "review" if self._should_review(graph_state) else "answer"

    def _after_review(self, graph_state: GraphState) -> str:
        state = self._load_state(graph_state)
        return "fail" if state.status == "failed" else "answer"

    def _after_answer(self, graph_state: GraphState) -> str:
        state = self._load_state(graph_state)
        return "fail" if state.status == "failed" else "audit"

    def _after_audit(self, graph_state: GraphState) -> str:
        state = self._load_state(graph_state)
        return "fail" if state.status == "failed" else "finish"

    def _register_tools(self) -> None:
        self.sandbox.register("route_query", tool_functions.route_query)
        self.sandbox.register("retrieve_evidence", tool_functions.retrieve_evidence)
        self.sandbox.register("analyze_real_window", tool_functions.analyze_real_window)
        self.sandbox.register("generate_answer", tool_functions.generate_answer)
        self.sandbox.register("evaluate_answer", tool_functions.evaluate_answer)

    def _build_checkpointer(self):
        if not self.enable_checkpoint:
            self.checkpoint_backend = "none"
            return None
        if self.checkpoint_backend == "memory":
            return InMemorySaver()
        if self.checkpoint_backend == "sqlite":
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "SQLite checkpoint backend requires `langgraph-checkpoint-sqlite`. "
                    "Install it with `pip install langgraph-checkpoint-sqlite`."
                ) from exc
            if self.checkpoint_path is None:
                self.checkpoint_path = self.store.root / "langgraph_checkpoints.sqlite"
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_context = SqliteSaver.from_conn_string(str(self.checkpoint_path))
            saver = self._checkpoint_context.__enter__()
            if hasattr(saver, "setup"):
                saver.setup()
            return saver
        raise ValueError(f"Unsupported checkpoint_backend={self.checkpoint_backend}")

    def _record_checkpoint_event(self, state: AgentState, thread_id: str) -> None:
        if not self.enable_checkpoint:
            return
        self.store.save_state(state)
        self.store.append_event(
            state.run_id,
            "langgraph_checkpoint_enabled",
            {
                "status": state.status,
                "thread_id": thread_id,
                "backend": self.checkpoint_backend,
                "requested_backend": self.requested_checkpoint_backend,
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "notice": self.checkpoint_notice,
            },
        )

    def _review_payload(self, state: AgentState) -> Dict[str, Any]:
        top_evidence = []
        fact_card_preview = ""
        for item in state.retrieved:
            if item.get("citation_id") == "S9001":
                fact_card_preview = str(item.get("text", ""))[:300]
                continue
            if len(top_evidence) >= 5:
                continue
            top_evidence.append(
                {
                    "citation_id": item.get("citation_id"),
                    "source_type": item.get("source_type"),
                    "metadata": dict(item.get("metadata") or {}),
                    "text": str(item.get("text", ""))[:200],
                }
            )
        if not fact_card_preview and state.fact_card_text:
            fact_card_preview = state.fact_card_text[:300]
        return {
            "question": state.question,
            "route": state.route.model_dump(mode="json") if state.route else None,
            "top_retrieved_evidence": top_evidence,
            "fact_card_preview": fact_card_preview,
            "instruction": "Approve, edit, or cancel before answer generation.",
            "allowed_actions": ["approve", "drop_citations", "edit_question", "cancel"],
        }

    def _normalize_review_decision(self, decision: dict[str, Any] | bool | str | Any) -> Dict[str, Any]:
        if isinstance(decision, dict):
            action = str(decision.get("action", "approve"))
            return {**decision, "action": action}
        if isinstance(decision, bool):
            return {"action": "approve" if decision else "cancel", "reason": "Boolean human review response."}
        if isinstance(decision, str):
            return {"action": decision}
        return {"action": "approve"}

    def _to_run_result(self, result: Dict[str, Any], thread_id: str, config: Dict[str, Any]) -> LangGraphRunResult:
        if "state" in result:
            state = AgentState.model_validate(result["state"])
        else:
            snapshot = self.graph.get_state(config)
            if not snapshot.values or "state" not in snapshot.values:
                raise ValueError(f"No state found for thread_id={thread_id}")
            state = AgentState.model_validate(snapshot.values["state"])
        interrupts = self._serialize_interrupts(result.get("__interrupt__", []))
        return LangGraphRunResult(
            state=state,
            thread_id=thread_id,
            interrupted=bool(interrupts),
            interrupts=interrupts,
            output_dir=str(self.store.run_dir(state.run_id)),
        )

    def _serialize_interrupts(self, interrupts: Any) -> list[Dict[str, Any]]:
        serialized = []
        for item in interrupts or []:
            serialized.append(
                {
                    "id": getattr(item, "id", None),
                    "value": getattr(item, "value", item),
                }
            )
        return serialized

    def _thread_config(self, thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _should_review(self, graph_state: GraphState) -> bool:
        return bool(self._options(graph_state).get("human_review", False))

    def _options(self, graph_state: GraphState) -> Dict[str, Any]:
        return dict(graph_state.get("options") or {})

    def _load_state(self, graph_state: GraphState) -> AgentState:
        return AgentState.model_validate(graph_state["state"])

    def _dump_state(self, state: AgentState, graph_state: GraphState | None = None) -> GraphState:
        result: GraphState = {"state": state.model_dump(mode="json")}
        if graph_state and "options" in graph_state:
            result["options"] = dict(graph_state["options"])
        return result
