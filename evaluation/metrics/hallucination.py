from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from core.schema import RetrievedChunk
from core.text import keyword_overlap
from llm_interface.response_parser import extract_citations, split_claims


@dataclass
class HallucinationReport:
    claim_count: int
    supported_claims: int
    unsupported_claims: List[str]
    hallucination_rate: float
    citation_exists_claims: int = 0
    citation_weak_claims: List[str] = field(default_factory=list)
    citation_support_rate: float = 0.0
    support_threshold: float = 0.18


def hallucination_rate(
    answer: str,
    retrieved: Iterable[RetrievedChunk],
    *,
    support_threshold: float = 0.18,
) -> HallucinationReport:
    claims = [claim for claim in split_claims(answer) if len(claim) >= 8]
    hits = list(retrieved)
    citation_map = {hit.citation_id: hit for hit in hits}
    unsupported: List[str] = []
    citation_weak: List[str] = []
    supported = 0
    citation_exists_claims = 0
    citation_supported_claims = 0

    for claim in claims:
        citations = extract_citations(claim)
        if citations:
            cited_hits = [citation_map[cid] for cid in citations if cid in citation_map]
            if len(cited_hits) != len(citations):
                unsupported.append(claim)
                continue

            citation_exists_claims += 1
            best_cited_overlap = max((keyword_overlap(claim, hit.chunk.text) for hit in cited_hits), default=0.0)
            if best_cited_overlap >= support_threshold:
                supported += 1
                citation_supported_claims += 1
            else:
                citation_weak.append(claim)
                unsupported.append(claim)
            continue

        # Fallback for uncited claims: strong lexical overlap with any retrieved evidence
        # is treated as supported, but it does not improve citation support rate.
        best = max((keyword_overlap(claim, hit.chunk.text) for hit in hits), default=0.0)
        if best >= support_threshold:
            supported += 1
        else:
            unsupported.append(claim)

    total = len(claims)
    return HallucinationReport(
        claim_count=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        hallucination_rate=(len(unsupported) / total) if total else 0.0,
        citation_exists_claims=citation_exists_claims,
        citation_weak_claims=citation_weak,
        citation_support_rate=(citation_supported_claims / citation_exists_claims) if citation_exists_claims else 0.0,
        support_threshold=support_threshold,
    )
