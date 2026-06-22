from __future__ import annotations

import re
from typing import Dict, List, Optional


KNOWN_MODELS = [
    "Anomaly Transformer",
    "DCdetector",
    "DADA",
    "CATCH",
    "MtsCID",
    "TimeMixer",
    "iTransformer",
]

KNOWN_DATASETS = [
    "SMD",
    "MSL",
    "SMAP",
    "PSM",
    "SWAT",
    "GECCO",
    "Creditcard",
    "CICIDS",
    "NIPS_TS_Water",
    "NIPS_TS_Swan",
]


def _ordered_unique(matches: List[tuple[int, str]]) -> List[str]:
    seen = set()
    values: List[str] = []
    for _, value in sorted(matches, key=lambda item: item[0]):
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def infer_filter_lists(query: str) -> Dict[str, List[str]]:
    lower = query.lower()
    model_matches: List[tuple[int, str]] = []
    dataset_matches: List[tuple[int, str]] = []
    for name in KNOWN_MODELS:
        index = lower.find(name.lower())
        if index >= 0:
            model_matches.append((index, name))
    for name in KNOWN_DATASETS:
        index = lower.find(name.lower())
        if index >= 0:
            dataset_matches.append((index, name))
    asd = re.search(r"ASD[_ -]?dataset[_ -]?(\d+)|ASD[_ -]?(\d+)", query, re.I)
    if asd:
        dataset_matches.append((asd.start(), f"ASD_dataset_{asd.group(1) or asd.group(2)}"))
    return {
        "model_names": _ordered_unique(model_matches),
        "dataset_names": _ordered_unique(dataset_matches),
    }


def infer_filters(query: str) -> Dict[str, Optional[str]]:
    lists = infer_filter_lists(query)
    model_names = lists["model_names"]
    dataset_names = lists["dataset_names"]
    return {
        "model_name": model_names[0] if model_names else None,
        "dataset_name": dataset_names[0] if dataset_names else None,
    }


def rewrite_query(query: str) -> str:
    synonyms = {
        "为什么": "原因 explanation",
        "贡献最大": "attribution contribution variable",
        "表现差": "low performance poor result metric",
        "窗口": "window segment time-series anomaly",
        "F1": "F-score f1 Affiliation_F1",
    }
    rewritten = query
    for key, value in synonyms.items():
        if key in query:
            rewritten += " " + value
    return rewritten
