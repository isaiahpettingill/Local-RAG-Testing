from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import ContextManager

from src.models.config import KNOWLEDGEBASE_DB, EVAL_QUEUE_DB, INGESTION_QUEUE_DB  # noqa: F401


class SQLiteConnection(ContextManager):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def __exit__(self, *args: object) -> None:
        if self.conn:
            self.conn.close()


def get_knowledgebase_conn() -> SQLiteConnection:
    return SQLiteConnection(KNOWLEDGEBASE_DB)


def get_eval_queue_conn() -> SQLiteConnection:
    return SQLiteConnection(EVAL_QUEUE_DB)


def get_ingestion_queue_conn() -> SQLiteConnection:
    return SQLiteConnection(INGESTION_QUEUE_DB)


def get_staging_conn() -> SQLiteConnection:
    from src.models.config import STAGING_DB

    return SQLiteConnection(STAGING_DB)


def get_ladybug_conn():
    from src.models.config import LADYBUGDB_DIR
    from real_ladybug import Database
    from real_ladybug.connection import Connection

    db = Database(LADYBUGDB_DIR.as_posix())
    return Connection(db)
