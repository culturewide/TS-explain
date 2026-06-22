from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    raw: Dict[str, object] = field(default_factory=dict)

