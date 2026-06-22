from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from core.schema import RetrievedChunk
from core.text import keyword_overlap
from llm_interface.response_parser import extract_citations, split_claims


@dataclass
class CitationCheck:
    citation_id: str
    exists: bool
    overlap: float
    entailment_hint: str


def citation_fidelity(answer: str, retrieved: Iterable[RetrievedChunk], *, threshold: float = 0.12) -> List[CitationCheck]:
    citation_map = {hit.citation_id: hit for hit in retrieved}
    checks: List[CitationCheck] = []
    for claim in split_claims(answer):
        for cid in extract_citations(claim):
            hit = citation_map.get(cid)
            if hit is None:
                checks.append(CitationCheck(cid, False, 0.0, "引用编号不存在于本次检索结果"))
                continue
            overlap = keyword_overlap(claim, hit.chunk.text)
            hint = "词面支撑较充分，可进入人工/NLI复核" if overlap >= threshold else "词面支撑偏弱，需要人工/NLI复核"
            checks.append(CitationCheck(cid, True, overlap, hint))
    return checks

