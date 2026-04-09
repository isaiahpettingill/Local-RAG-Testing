from __future__ import annotations

from typing import Any

from src.db.connection import (
    get_eval_queue_conn,
    get_ingestion_queue_conn,
    get_knowledgebase_conn,
)


def create_knowledgebase_schema() -> None:
    with get_knowledgebase_conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, text TEXT, vector BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id')"
        )
        conn.commit()


def create_ingestion_queue_schema() -> None:
    with get_ingestion_queue_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_queue (
                chunk_id INTEGER PRIMARY KEY,
                raw_text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                graph_extraction_attempts INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def create_eval_queue_schema() -> None:
    with get_eval_queue_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_queue (
                eval_id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                ground_truth TEXT NOT NULL,
                pipeline_a_status TEXT NOT NULL DEFAULT 'PENDING',
                pipeline_b_status TEXT NOT NULL DEFAULT 'PENDING',
                pipeline_c_status TEXT NOT NULL DEFAULT 'PENDING',
                pipeline_a_result TEXT,
                pipeline_b_result TEXT,
                pipeline_c_result TEXT,
                grader_status TEXT NOT NULL DEFAULT 'PENDING',
                grader_score_a REAL,
                grader_score_b REAL,
                grader_score_c REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def init_all_schemas() -> None:
    create_knowledgebase_schema()
    create_ingestion_queue_schema()
    create_eval_queue_schema()
