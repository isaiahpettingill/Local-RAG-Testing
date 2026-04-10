from __future__ import annotations

import logging
from typing import Any

from src.db.connection import get_ladybug_conn

log = logging.getLogger(__name__)


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _sanitize_label(value: Any) -> str:
    label = str(value).strip() or "concept"
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in label)
    return cleaned or "concept"


def _graph_store() -> Any | None:
    try:
        import real_ladybug as lb
        from llama_index.graph_stores.ladybug import LadybugGraphStore
    except Exception:
        return None

    from src.models.config import LADYBUGDB_DIR

    db = lb.Database(LADYBUGDB_DIR.as_posix())
    return LadybugGraphStore(db)


def insert_graph_data(
    graph: dict[str, list[dict[str, Any]]],
    source_url: str | None = None,
    source_title: str | None = None,
    chunk_id: int | None = None,
) -> None:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    store = _graph_store()
    if store is not None:
        for edge in edges:
            source = str(edge.get("source", "")).strip()
            relation = str(edge.get("type", "related_to")).strip() or "related_to"
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue
            store.upsert_triplet(source, relation, target)
        return

    with get_ladybug_conn() as conn:
        for node in nodes:
            name = str(node.get("name", "")).strip()
            if not name:
                continue
            label = _sanitize_label(node.get("type", "concept"))
            conn.execute(
                f"MERGE (n:{label} {{name: '{_escape(name)}'}})"
            )
            if source_url or source_title or chunk_id is not None:
                props: dict[str, Any] = {}
                if source_url:
                    props["source_url"] = source_url
                if source_title:
                    props["source_title"] = source_title
                if chunk_id is not None:
                    props["chunk_id"] = chunk_id
                prop_clause = ", ".join(
                    f"{key}: '{_escape(value)}'" for key, value in props.items()
                )
                conn.execute(
                    f"MATCH (n:{label} {{name: '{_escape(name)}'}}) SET n += {{{prop_clause}}}"
                )

        for edge in edges:
            source = str(edge.get("source", "")).strip()
            relation = _sanitize_label(edge.get("type", "related_to"))
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue
            conn.execute(
                f"MATCH (a {{name: '{_escape(source)}'}}), (b {{name: '{_escape(target)}'}}) MERGE (a)-[:{relation}]->(b)"
            )
        conn.commit()


def _format_relationship_result(result: Any, index: int) -> str:
    if isinstance(result, dict):
        source = result.get("source") or result.get("from") or result.get("subject") or result.get("name") or index
        relation = result.get("relation") or result.get("type") or result.get("predicate") or "related_to"
        target = result.get("target") or result.get("to") or result.get("object") or result.get("value") or ""
        text = f"{source} -[{relation}]-> {target}".strip()
    else:
        source = getattr(result, "source", None) or getattr(result, "from_entity", None) or getattr(result, "subject", None) or getattr(result, "name", None) or index
        relation = getattr(result, "relation", None) or getattr(result, "type", None) or getattr(result, "predicate", None) or "related_to"
        target = getattr(result, "target", None) or getattr(result, "to_entity", None) or getattr(result, "object", None) or getattr(result, "value", None) or ""
        text = f"{source} -[{relation}]-> {target}".strip()
    return f"[Source: {source}] {text}"


def get_entity_relationships(entity_name: str, limit: int = 5) -> str:
    store = _graph_store()
    if store is not None:
        try:
            results = store.get(entity_name)
        except Exception as exc:
            log.warning("Ladybug graph lookup failed for %s: %s", entity_name, exc)
            results = []
        return "\n".join(
            _format_relationship_result(result, index)
            for index, result in enumerate(results[:limit], start=1)
        )

    escaped = _escape(entity_name)
    query = (
        "MATCH (a)-[r]->(b) "
        f"WHERE a.name CONTAINS '{escaped}' OR b.name CONTAINS '{escaped}' "
        "RETURN a.name AS source, type(r) AS relation, b.name AS target "
        f"LIMIT {int(limit)}"
    )
    with get_ladybug_conn() as conn:
        try:
            rows = conn.execute(query).fetchall()
        except Exception as exc:
            log.warning("Ladybug fallback query failed for %s: %s", entity_name, exc)
            rows = []
    return "\n".join(
        _format_relationship_result(dict(row), index)
        for index, row in enumerate(rows, start=1)
    )
