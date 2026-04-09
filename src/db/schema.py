from __future__ import annotations


from src.db.connection import (
    get_eval_queue_conn,
    get_ingestion_queue_conn,
    get_knowledgebase_conn,
)


def create_knowledgebase_schema() -> None:
    with get_knowledgebase_conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, text TEXT, vector BLOB, url TEXT)"
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
                url TEXT,
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
    create_staging_schema()


def create_staging_schema() -> None:
    from src.db.connection import get_staging_conn

    with get_staging_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS staging_pages (
                page_id INTEGER PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                text TEXT,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS staging_edges (
                edge_id INTEGER PRIMARY KEY,
                from_url TEXT NOT NULL,
                to_url TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS staging_entities (
                entity_id INTEGER PRIMARY KEY,
                page_id INTEGER NOT NULL,
                entity_name TEXT NOT NULL,
                entity_type TEXT,
                properties TEXT,
                source_page TEXT,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (page_id) REFERENCES staging_pages(page_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS staging_entity_edges (
                edge_id INTEGER PRIMARY KEY,
                from_entity TEXT NOT NULL,
                to_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                source_page TEXT,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crawl_state (
                url TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'DISCOVERED',
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                visited_at TIMESTAMP
            )
        """)
        conn.commit()
