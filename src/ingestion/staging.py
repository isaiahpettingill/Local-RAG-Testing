from __future__ import annotations

import json
import logging
from pathlib import Path

from src.db.connection import get_staging_conn

log = logging.getLogger(__name__)

CRAWL_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "crawl"
PAGES_FILE = CRAWL_DATA_DIR / "pages.jsonl"
GRAPH_FILE = CRAWL_DATA_DIR / "graph_edges.jsonl"
ENTITIES_FILE = CRAWL_DATA_DIR / "entities.jsonl"
RELATIONS_FILE = CRAWL_DATA_DIR / "relations.jsonl"


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


def stage_entities(entities_data: list[dict]) -> int:
    count = 0
    with get_staging_conn() as conn:
        for item in entities_data:
            page_url = item.get("page_url", "")
            page_row = conn.execute(
                "SELECT page_id FROM staging_pages WHERE url = ?", (page_url,)
            ).fetchone()
            if not page_row:
                continue
            page_id = page_row["page_id"]
            entities = item.get("entities", [])
            for ent in entities:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO staging_entities (page_id, entity_name, entity_type, properties, source_page, status)
                    VALUES (?, ?, ?, ?, ?, 'PENDING')
                    """,
                    (
                        page_id,
                        ent.get("name", ""),
                        ent.get("type", ""),
                        json.dumps(ent.get("properties", {})),
                        page_url,
                    ),
                )
                count += 1
        conn.commit()

    log.info("Staged %d entities", count)
    return count


def stage_entity_edges(relations_data: list[dict]) -> int:
    count = 0
    with get_staging_conn() as conn:
        for rel in relations_data:
            conn.execute(
                """
                INSERT OR IGNORE INTO staging_entity_edges (from_entity, to_entity, relation_type, source_page, status)
                VALUES (?, ?, ?, ?, 'PENDING')
                """,
                (
                    rel.get("from_entity", ""),
                    rel.get("to_entity", ""),
                    rel.get("relation_type", "related_to"),
                    rel.get("source_page", ""),
                ),
            )
            count += 1
        conn.commit()

    log.info("Staged %d entity edges", count)
    return count


def load_staging_to_ingestion() -> int:
    from src.db.connection import get_staging_conn, get_ingestion_queue_conn

    count = 0
    with get_staging_conn() as staging, get_ingestion_queue_conn() as queue:
        rows = staging.execute(
            "SELECT page_id, url, title, text FROM staging_pages WHERE status = 'PENDING' ORDER BY page_id"
        ).fetchall()
        for row in rows:
            queue.execute(
                """
                INSERT OR IGNORE INTO ingestion_queue (raw_text, url, source_title, status)
                VALUES (?, ?, ?, 'PENDING')
                """,
                (row["text"], row["url"], row["title"] or ""),
            )
            count += 1
        queue.commit()
    log.info("Loaded %d pages to ingestion queue", count)
    return count


def get_outgoing_links(page_url: str, limit: int = 25) -> list[str]:
    with get_staging_conn() as conn:
        rows = conn.execute(
            """
            SELECT to_url
            FROM staging_edges
            WHERE from_url = ?
            ORDER BY edge_id
            LIMIT ?
            """,
            (page_url, limit),
        ).fetchall()
        return [row["to_url"] for row in rows]


def get_unprocessed_pages(limit: int = 10) -> list[dict]:
    with get_staging_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.page_id, p.url, p.title, p.text, e.to_url as link
            FROM staging_pages p
            LEFT JOIN staging_edges e ON p.url = e.from_url
            WHERE p.status = 'PENDING'
            ORDER BY p.page_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def mark_page_extracted(page_id: int) -> None:
    with get_staging_conn() as conn:
        conn.execute(
            "UPDATE staging_pages SET status = 'EXTRACTED' WHERE page_id = ?",
            (page_id,),
        )
        conn.commit()
