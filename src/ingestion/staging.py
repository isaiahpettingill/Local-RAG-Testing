from __future__ import annotations

import json
import logging
from pathlib import Path

from src.db.connection import get_staging_conn

log = logging.getLogger(__name__)

CRAWL_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "crawl"
PAGES_FILE = CRAWL_DATA_DIR / "pages.jsonl"
GRAPH_FILE = CRAWL_DATA_DIR / "graph_edges.jsonl"


def stage_pages() -> int:
    count = 0
    if not PAGES_FILE.exists():
        log.warning("No pages file found at %s", PAGES_FILE)
        return 0

    with get_staging_conn() as conn:
        for line in PAGES_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            url = data["url"]
            title = data.get("title", "")
            text = data.get("text", "")
            conn.execute(
                """
                INSERT OR IGNORE INTO staging_pages (url, title, text, status)
                VALUES (?, ?, ?, 'PENDING')
                """,
                (url, title, text),
            )
            count += 1
        conn.commit()

    log.info("Staged %d pages", count)
    return count


def stage_edges() -> int:
    count = 0
    if not GRAPH_FILE.exists():
        log.warning("No graph file found at %s", GRAPH_FILE)
        return 0

    seen: set[tuple[str, str]] = set()
    with get_staging_conn() as conn:
        for line in GRAPH_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            from_url = data["from"]
            to_url = data["to"]
            if (from_url, to_url) in seen:
                continue
            seen.add((from_url, to_url))
            conn.execute(
                """
                INSERT OR IGNORE INTO staging_edges (from_url, to_url, status)
                VALUES (?, ?, 'PENDING')
                """,
                (from_url, to_url),
            )
            count += 1
        conn.commit()

    log.info("Staged %d edges", count)
    return count


def load_staging_to_ingestion() -> int:
    from src.db.connection import get_staging_conn, get_ingestion_queue_conn

    count = 0
    with get_staging_conn() as staging, get_ingestion_queue_conn() as queue:
        rows = staging.execute(
            "SELECT page_id, text FROM staging_pages WHERE status = 'PENDING' ORDER BY page_id"
        ).fetchall()
        for row in rows:
            queue.execute(
                """
                INSERT OR IGNORE INTO ingestion_queue (raw_text, status)
                VALUES (?, 'PENDING')
                """,
                (row["text"],),
            )
            count += 1
        queue.commit()
    log.info("Loaded %d pages to ingestion queue", count)
    return count
