from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from agent_runtime.schema import AgentState, RunRecord, model_to_dict, utc_now


class JsonlRunStore:
    """Append-only JSONL persistence for agent runs."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append_event(self, run_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        path = self.run_dir(run_id) / "events.jsonl"
        record = {"event_type": event_type, "created_at": utc_now(), **payload}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_state(self, state: AgentState) -> None:
        path = self.run_dir(state.run_id) / "state.json"
        path.write_text(json.dumps(model_to_dict(state), ensure_ascii=False, indent=2), encoding="utf-8")

    def save_final(self, record: RunRecord) -> None:
        run_dir = self.run_dir(record.run_id)
        (run_dir / "run_record.json").write_text(
            json.dumps(model_to_dict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index_path = self.root / "runs.jsonl"
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(model_to_dict(record), ensure_ascii=False) + "\n")

    def load_state(self, run_id: str) -> AgentState:
        path = self.run_dir(run_id) / "state.json"
        return AgentState.model_validate_json(path.read_text(encoding="utf-8"))

    def iter_run_records(self) -> Iterable[Dict[str, Any]]:
        index_path = self.root / "runs.jsonl"
        if not index_path.exists():
            return []
        return [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]

