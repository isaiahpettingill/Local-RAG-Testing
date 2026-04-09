from __future__ import annotations

import struct
from typing import Any

from src.db.connection import get_knowledgebase_conn


def insert_chunk(text: str, vector: list[float], url: str) -> int:
    with get_knowledgebase_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO chunks (text, vector, url) VALUES (?, ?, ?)",
            (text, _pack_vector(vector), url),
        )
        chunk_id_raw = cursor.lastrowid
        assert chunk_id_raw is not None
        chunk_id: int = chunk_id_raw
        conn.execute(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (chunk_id, text)
        )
        conn.commit()
        return chunk_id


def execute_fts(query: str, limit: int = 5) -> list[dict[str, Any]]:
    with get_knowledgebase_conn() as conn:
        rows = conn.execute(
            "SELECT rowid, text, url FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (query, limit),
        ).fetchall()
        return [{"id": r["rowid"], "text": r["text"], "url": r["url"]} for r in rows]


def execute_vector(query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
    packed = _pack_vector(query_vector)
    with get_knowledgebase_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT rowid, text, url FROM chunks_vec WHERE embedding MATCH ? LIMIT ?",
                (packed, limit),
            ).fetchall()
        except Exception:
            rows = conn.execute(
                "SELECT rowid, text, url FROM chunks WHERE rowid IN (SELECT rowid FROM chunks LIMIT ?)",
                (limit,),
            ).fetchall()
        return [{"id": r["rowid"], "text": r["text"], "url": r["url"]} for r in rows]


def reciprocal_rank_fusion(
    results_a: list[dict[str, Any]], results_b: list[dict[str, Any]], limit: int = 3
) -> list[dict[str, Any]]:
    scores: dict[int, float] = {}
    for rank, item in enumerate(results_a):
        scores[item["id"]] = scores.get(item["id"], 0) + 1 / (rank + 60)
    for rank, item in enumerate(results_b):
        scores[item["id"]] = scores.get(item["id"], 0) + 1 / (rank + 60)
    sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
    id_to_text = {r["id"]: r["text"] for r in results_a + results_b}
    id_to_url = {r["id"]: r["url"] for r in results_a + results_b}
    return [
        {"id": i, "text": id_to_text[i], "url": id_to_url[i]}
        for i in sorted_ids[:limit]
    ]


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)
