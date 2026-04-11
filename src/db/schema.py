from __future__ import annotations


from src.db.connection import (
    get_crawl_conn,
    get_eval_queue_conn,
    get_ingestion_queue_conn,
    get_knowledgebase_conn,
)


def _ensure_columns(conn, table: str, columns: dict[str, str]) -> None:
    existing = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def create_knowledgebase_schema() -> None:
    with get_knowledgebase_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT,
                raw_text TEXT NOT NULL DEFAULT '',
                context TEXT NOT NULL DEFAULT '',
                vector BLOB,
                url TEXT NOT NULL DEFAULT '',
                source_title TEXT NOT NULL DEFAULT '',
                chunk_index INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id')"
        )
        _ensure_columns(
            conn,
            "chunks",
            {
                "raw_text": "TEXT NOT NULL DEFAULT ''",
                "context": "TEXT NOT NULL DEFAULT ''",
                "url": "TEXT NOT NULL DEFAULT ''",
                "source_title": "TEXT NOT NULL DEFAULT ''",
                "chunk_index": "INTEGER NOT NULL DEFAULT 0",
            },
        )
        conn.commit()


def create_ingestion_queue_schema() -> None:
    with get_ingestion_queue_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_queue (
                chunk_id INTEGER PRIMARY KEY,
                staging_page_id INTEGER,
                raw_text TEXT NOT NULL DEFAULT '',
                url TEXT,
                source_title TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'PENDING',
                graph_extraction_attempts INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _ensure_columns(
            conn,
            "ingestion_queue",
            {
                "staging_page_id": "INTEGER",
                "raw_text": "TEXT NOT NULL DEFAULT ''",
                "url": "TEXT",
                "source_title": "TEXT NOT NULL DEFAULT ''",
            },
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_queue_status_chunk ON ingestion_queue(status, chunk_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_queue_staging_page_id ON ingestion_queue(staging_page_id)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ingestion_queue_unique_staging_page ON ingestion_queue(staging_page_id) WHERE staging_page_id IS NOT NULL"
        )
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
    create_crawl_schema()
    migrate_crawl_state()


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
        conn.commit()


def create_crawl_schema() -> None:
    with get_crawl_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crawl_state (
                url TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'DISCOVERED',
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                visited_at TIMESTAMP
            )
        """)
        conn.commit()


def migrate_crawl_state() -> None:
    from src.db.connection import get_staging_conn

    with get_staging_conn() as staging, get_crawl_conn() as crawl:
        staging_has_table = staging.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'crawl_state'"
        ).fetchone()
        if not staging_has_table:
            return

        rows = staging.execute(
            "SELECT url, status, discovered_at, visited_at FROM crawl_state ORDER BY url"
        ).fetchall()
        for row in rows:
            crawl.execute(
                """
                INSERT OR IGNORE INTO crawl_state (url, status, discovered_at, visited_at)
                VALUES (?, ?, ?, ?)
                """,
                (row["url"], row["status"], row["discovered_at"], row["visited_at"]),
            )
        crawl.commit()
