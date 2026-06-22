from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from core.schema import KnowledgeChunk


class MetadataStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                citation_id TEXT,
                source_type TEXT,
                source_path TEXT,
                source_id TEXT,
                model_name TEXT,
                dataset_name TEXT,
                metadata_json TEXT
            )
            """
        )
        self.conn.commit()

    def reset(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    def upsert_chunks(self, chunks: Iterable[KnowledgeChunk]) -> None:
        rows = []
        for chunk in chunks:
            meta = chunk.metadata
            rows.append(
                (
                    chunk.id,
                    meta.get("citation_id"),
                    chunk.source_type,
                    chunk.source_path,
                    chunk.source_id,
                    meta.get("model_name"),
                    meta.get("dataset_name"),
                    json.dumps(meta, ensure_ascii=False),
                )
            )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (id, citation_id, source_type, source_path, source_id, model_name, dataset_name, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def find_ids(
        self,
        *,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[str]:
        clauses = []
        params = []
        if model_name:
            clauses.append("lower(model_name)=lower(?)")
            params.append(model_name)
        if dataset_name:
            clauses.append("lower(dataset_name)=lower(?)")
            params.append(dataset_name)
        if source_type:
            clauses.append("source_type=?")
            params.append(source_type)
        sql = "SELECT id FROM chunks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [row[0] for row in self.conn.execute(sql, params)]

    def close(self) -> None:
        self.conn.close()

