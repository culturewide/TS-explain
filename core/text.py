from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+(?:[-.][A-Za-z0-9_]+)*")


def normalize_text(text: object) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(normalize_text(text))]


def chunk_by_paragraphs(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 120,
    min_chars: int = 80,
) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            if len(current) >= min_chars:
                chunks.append(current)
            overlap = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{overlap}\n\n{paragraph}" if overlap else paragraph
    if len(current) >= min_chars:
        chunks.append(current)
    if not chunks and text.strip():
        chunks.append(text.strip()[:max_chars])
    return chunks


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def hashing_embedding(text: str, dim: int = 384) -> List[float]:
    """Deterministic local embedding for offline builds and tests.

    This is not a semantic model. It gives the project a reproducible local
    vector backend until OpenAI/BGE embeddings are configured.
    """

    vector = [0.0] * dim
    for token, count in Counter(tokenize(text)).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        idx = raw % dim
        sign = 1.0 if (raw >> 8) & 1 else -1.0
        vector[idx] += sign * (1.0 + math.log1p(count))
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    na = math.sqrt(sum(float(x) * float(x) for x in a[:n])) or 1.0
    nb = math.sqrt(sum(float(x) * float(x) for x in b[:n])) or 1.0
    return dot / (na * nb)


def keyword_overlap(query: str, document: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    d = set(tokenize(document))
    return len(q & d) / len(q)


def bm25_scores(query: str, documents: Iterable[str], k1: float = 1.5, b: float = 0.75) -> Dict[int, float]:
    docs = [tokenize(doc) for doc in documents]
    query_terms = tokenize(query)
    if not docs or not query_terms:
        return {}
    avgdl = sum(len(doc) for doc in docs) / max(len(docs), 1)
    df: Counter[str] = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1
    scores: Dict[int, float] = {}
    n_docs = len(docs)
    for idx, doc in enumerate(docs):
        counts = Counter(doc)
        dl = len(doc) or 1
        score = 0.0
        for term in query_terms:
            if counts[term] == 0:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            tf = counts[term]
            denom = tf + k1 * (1 - b + b * dl / (avgdl or 1))
            score += idf * (tf * (k1 + 1)) / denom
        scores[idx] = score
    max_score = max(scores.values()) if scores else 0.0
    if max_score > 0:
        scores = {idx: score / max_score for idx, score in scores.items()}
    return scores

