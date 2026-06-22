from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


Metadata = Dict[str, Any]


@dataclass(slots=True)
class KnowledgeChunk:
    """A traceable unit stored in the knowledge base."""

    id: str
    text: str
    source_type: str
    source_path: str
    source_id: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """A knowledge chunk plus retrieval scores."""

    chunk: KnowledgeChunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: Optional[float] = None

    @property
    def citation_id(self) -> str:
        return self.chunk.metadata.get("citation_id", self.chunk.id)


@dataclass(slots=True)
class RetrievalQuery:
    query: str
    top_k: int = 8
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    source_types: Optional[List[str]] = None
    mode: str = "hybrid"


@dataclass(slots=True)
class TrendFact:
    variable: str
    tau: float
    direction: str
    strength: str
    stl_trend_delta: Optional[float] = None


@dataclass(slots=True)
class PeriodicityFact:
    variable: str
    dominant_period: Optional[float]
    fft_power_ratio: float
    acf_peak_lag: Optional[int]
    significant: bool


@dataclass(slots=True)
class CorrelationFact:
    variable_a: str
    variable_b: str
    pearson: float
    spearman: Optional[float] = None
    changed: bool = False


@dataclass(slots=True)
class AnomalyProfile:
    density: float
    cluster_count: int
    peak_position: Optional[int]
    temporal_distribution: str
    score_summary: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ResidualFact:
    variable: str
    mean_bias: float
    std: float
    max_abs_error: float
    amplification_regions: List[str] = field(default_factory=list)


@dataclass(slots=True)
class AttributionFact:
    variable: str
    contribution: float
    reason: str


@dataclass(slots=True)
class TSFactCard:
    """Structured time-series evidence for prompt injection."""

    window_id: str
    length: int
    variables: List[str]
    trends: List[TrendFact] = field(default_factory=list)
    periodicities: List[PeriodicityFact] = field(default_factory=list)
    correlations: List[CorrelationFact] = field(default_factory=list)
    anomaly_profile: Optional[AnomalyProfile] = None
    residuals: List[ResidualFact] = field(default_factory=list)
    attributions: List[AttributionFact] = field(default_factory=list)
    changepoints: List[int] = field(default_factory=list)
    stationarity: Metadata = field(default_factory=dict)
    narrative: str = ""


@dataclass(slots=True)
class ExplanationRequest:
    question: str
    mode: str = "rag_ts"
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    window_id: Optional[str] = None
    fact_card: Optional[TSFactCard] = None


@dataclass(slots=True)
class ExplanationAnswer:
    answer: str
    citations: List[str] = field(default_factory=list)
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    mode: str = "rag_ts"
    provider: str = "offline"
    raw: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class QuestionItem:
    id: str
    category: str
    question: str
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    expected_sources: List[str] = field(default_factory=list)

