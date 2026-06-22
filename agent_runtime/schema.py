from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


AgentName = Literal[
    "Supervisor",
    "QueryRouter",
    "EvidenceRetriever",
    "WindowAnalyzer",
    "AnswerWriter",
    "CitationAuditor",
]

StepStatus = Literal["planned", "running", "succeeded", "failed", "skipped"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"tool-{uuid4().hex[:10]}")
    agent: AgentName
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 60.0
    cwd: Optional[str] = None
    created_at: str = Field(default_factory=utc_now)


class Observation(BaseModel):
    tool_call_id: str
    agent: AgentName
    tool_name: str
    ok: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    elapsed_seconds: float = 0.0
    created_at: str = Field(default_factory=utc_now)


class AgentStep(BaseModel):
    index: int
    agent: AgentName
    phase: Literal["plan", "tool", "observe", "evaluate", "revise", "finish"]
    status: StepStatus = "planned"
    message: str = ""
    tool_call: Optional[ToolCall] = None
    observation: Optional[Observation] = None
    created_at: str = Field(default_factory=utc_now)


class RouteDecision(BaseModel):
    question_type: Literal["dataset_model", "window_explanation", "comparison", "unknown"] = "unknown"
    mode: Literal["no-rag", "rag-only", "rag-ts"] = "rag-only"
    needs_retrieval: bool = True
    needs_window: bool = False
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    model_names: List[str] = Field(default_factory=list)
    dataset_names: List[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = 1.0
    router_type: Literal["rule", "llm", "hybrid"] = "rule"
    uncertainty_reasons: List[str] = Field(default_factory=list)


class MemorySummary(BaseModel):
    run_id: str
    summary: str
    key_evidence: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class AgentState(BaseModel):
    run_id: str = Field(default_factory=new_run_id)
    question: str
    provider: str = "offline"
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    auto_window: bool = False
    window_length: int = 256
    window_strategy: str = "max_label_density"
    route: Optional[RouteDecision] = None
    retrieved: List[Dict[str, Any]] = Field(default_factory=list)
    fact_card_text: Optional[str] = None
    answer: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    steps: List[AgentStep] = Field(default_factory=list)
    memory_summary: Optional[MemorySummary] = None
    status: Literal["created", "running", "succeeded", "failed"] = "created"
    failures: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class RunRecord(BaseModel):
    run_id: str
    question: str
    provider: str
    status: str
    output_dir: str
    final_answer: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    memory_summary: Optional[MemorySummary] = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class HarnessItem(BaseModel):
    id: str
    question: str
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    auto_window: bool = False
    provider: Optional[str] = None


class HarnessResult(BaseModel):
    id: str
    run_id: Optional[str] = None
    status: Literal["pending", "succeeded", "failed", "skipped"] = "pending"
    attempt: int = 1
    previous_status: Optional[str] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    return model.model_dump(mode="json")


def ensure_relative_to(path: str | Path, root: str | Path) -> Path:
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path {resolved} is outside allowed root {root_resolved}") from exc
    return resolved
