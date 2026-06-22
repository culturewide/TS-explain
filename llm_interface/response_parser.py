from __future__ import annotations

import re
from typing import List


def extract_citations(text: str) -> List[str]:
    seen = []
    for citation in re.findall(r"\[(S\d{4})\]", text):
        if citation not in seen:
            seen.append(citation)
    return seen


def split_claims(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]
