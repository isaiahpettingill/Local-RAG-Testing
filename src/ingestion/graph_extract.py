from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.db.connection import get_ladybug_conn
from src.models.chat_llamacpp import ChatLlamaCpp

log = logging.getLogger(__name__)

NODE_TYPES = (
    "universe",
    "series",
    "book",
    "world",
    "character",
    "faction",
    "magic_system",
    "artifact",
    "event",
    "concept",
)

RELATION_TYPES = (
    "part_of_universe",
    "part_of_series",
    "appears_in_book",
    "set_on_world",
    "member_of",
    "leads",
    "ally_of",
    "enemy_of",
    "mentor_of",
    "student_of",
    "family_of",
    "uses_magic_system",
    "wields",
    "associated_with_artifact",
    "associated_with_event",
    "originates_from_world",
    "located_in",
    "same_as",
    "alias_of",
    "mentions",
    "related_to",
)

NODE_TYPE_ALIASES = {
    "person": "character",
    "people": "character",
    "place": "world",
    "location": "world",
    "organization": "faction",
    "org": "faction",
    "power": "magic_system",
    "ability": "magic_system",
    "item": "artifact",
    "object": "artifact",
    "concept": "concept",
    "event": "event",
    "series": "series",
    "book": "book",
    "novel": "book",
    "story": "book",
    "universe": "universe",
    "world": "world",
    "faction": "faction",
    "magic_system": "magic_system",
    "artifact": "artifact",
}

RELATION_TYPE_ALIASES = {
    "part_of": "part_of_series",
    "belongs_to": "part_of_universe",
    "in_universe": "part_of_universe",
    "appears_in": "appears_in_book",
    "set_in": "set_on_world",
    "member": "member_of",
    "leader_of": "leads",
    "friend_of": "ally_of",
    "rival_of": "enemy_of",
    "teacher_of": "mentor_of",
    "parent_of": "family_of",
    "child_of": "family_of",
    "uses": "uses_magic_system",
    "connected_to": "related_to",
    "references": "mentions",
    "alias": "alias_of",
    "aka": "alias_of",
    "same": "same_as",
}


@dataclass(slots=True)
class GraphNode:
    name: str
    node_type: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.node_type,
            "properties": self.properties,
        }


@dataclass(slots=True)
class GraphEdge:
    source: str
    relation: str
    target: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "type": self.relation,
            "target": self.target,
            "properties": self.properties,
        }


@dataclass(slots=True)
class GraphAlias:
    alias: str
    canonical: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "canonical": self.canonical,
            "properties": self.properties,
        }


def _normalize_node_type(value: Any) -> str:
    normalized = str(value).strip().lower().replace(" ", "_") or "concept"
    return NODE_TYPE_ALIASES.get(
        normalized, normalized if normalized in NODE_TYPES else "concept"
    )


def _normalize_relation_type(value: Any) -> str:
    normalized = str(value).strip().lower().replace(" ", "_") or "related_to"
    return RELATION_TYPE_ALIASES.get(
        normalized, normalized if normalized in RELATION_TYPES else "related_to"
    )


def _safe_title(text: str) -> str:
    return text.replace("\n", " ").strip()[:160]


def _guess_universe(title: str, text: str) -> str:
    haystack = f"{title} {text}".lower()
    for needle, universe in (
        ("cosmere", "Cosmere"),
        ("cytoverse", "Cytoverse"),
        ("reckoners", "Reckoners"),
        ("wheel of time", "Wheel of Time"),
        ("dark one", "Dark One"),
        ("standalone", "Standalone"),
    ):
        if needle in haystack:
            return universe
    return "Unknown"


def _infer_kind(title: str, text: str) -> str:
    title_l = title.lower()
    text_l = text.lower()
    if any(word in title_l for word in ("series", "saga", "cycle")):
        return "series"
    if any(
        word in title_l
        for word in ("chronicles", "archive", "trilogy", "novel", "novella", "story")
    ):
        return "book"
    if re.search(r"\b(character|person|people|born|father|mother|son|daughter)\b", text_l):
        return "character"
    if re.search(r"\b(world|planet|city|kingdom|continent|moon|planetary|realm)\b", text_l):
        return "world"
    if re.search(r"\b(concept|magic|power|ability|order|organization|species)\b", text_l):
        return "concept"
    return "concept"


def _build_ontology_prompt(
    text: str,
    title: str,
    page_url: str,
    outgoing_links: list[str],
    chunk_index: int,
    existing_relationships: str,
) -> str:
    links_str = "\n".join(outgoing_links[:20]) or "(none)"
    universe = _guess_universe(title, text)
    kind = _infer_kind(title, text)
    existing = existing_relationships.strip() or "(none)"

    return (
        "You are extracting a semantic knowledge graph from Coppermind pages about Brandon Sanderson's works.\n"
        "Use this ontology: Universe > Series > Book > World > Character > Faction > MagicSystem > Artifact > Event > Concept.\n"
        "Prefer high-precision nodes over exhaustive noise. Merge aliases when they clearly refer to the same thing.\n"
        "Return only JSON with nodes, edges, and aliases.\n\n"
        f"Page title: {title}\n"
        f"Page URL: {page_url}\n"
        f"Chunk index: {chunk_index}\n"
        f"Likely universe: {universe}\n"
        f"Likely page kind: {kind}\n"
        f"Outgoing links:\n{links_str}\n\n"
        f"Existing similar relationships already in the graph:\n{existing}\n\n"
        "Return JSON only with keys: nodes, edges, aliases.\n"
        "Each node must have: name, type, properties.\n"
        "Each edge must have: source, type, target, properties.\n"
        "Each alias must have: alias, canonical, properties.\n"
        "Useful edge types include: part_of_universe, part_of_series, appears_in_book, set_on_world, member_of, leads, ally_of, enemy_of, mentor_of, student_of, family_of, uses_magic_system, wields, associated_with_artifact, associated_with_event, originates_from_world, located_in, same_as, alias_of, mentions, related_to.\n\n"
        f"Text:\n{text[:6000]}\n"
    )


def _load_json_payload(result: str) -> dict[str, Any]:
    data = json.loads(result)
    return data if isinstance(data, dict) else {}


def _deduplicate_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[GraphEdge] = []
    for edge in edges:
        key = (edge.source.casefold(), edge.relation.casefold(), edge.target.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _fallback_graph(
    title: str, page_url: str, text: str, outgoing_links: list[str], chunk_index: int
) -> tuple[list[GraphNode], list[GraphEdge], list[GraphAlias]]:
    universe = _guess_universe(title, text)
    kind = _infer_kind(title, text)

    nodes = [
        GraphNode(
            name=_safe_title(title) or page_url,
            node_type=kind,
            properties={
                "source_url": page_url,
                "chunk_index": chunk_index,
                "universe": universe,
            },
        )
    ]
    if universe != "Unknown":
        nodes.append(
            GraphNode(
                name=universe,
                node_type="universe",
                properties={"source_url": page_url},
            )
        )

    edges: list[GraphEdge] = []
    if universe != "Unknown":
        edges.append(
            GraphEdge(
                source=_safe_title(title) or page_url,
                relation="part_of_universe",
                target=universe,
                properties={"source_url": page_url, "chunk_index": chunk_index},
            )
        )

    aliases: list[GraphAlias] = []
    for link in outgoing_links[:5]:
        edges.append(
            GraphEdge(
                source=_safe_title(title) or page_url,
                relation="mentions",
                target=link,
                properties={"source_url": page_url, "chunk_index": chunk_index},
            )
        )

    return nodes, edges, aliases


def extract_graph(
    text: str,
    title: str = "",
    page_url: str = "",
    outgoing_links: list[str] | None = None,
    chunk_index: int = 0,
    existing_relationships: str = "",
) -> dict[str, list[dict[str, Any]]]:
    extractor = ChatLlamaCpp("graph_summarizer")
    prompt = _build_ontology_prompt(
        text=text,
        title=title,
        page_url=page_url,
        outgoing_links=outgoing_links or [],
        chunk_index=chunk_index,
        existing_relationships=existing_relationships,
    )

    try:
        result = extractor.completion(prompt)
        data = _load_json_payload(result)
        nodes_raw = data.get("nodes", [])
        edges_raw = data.get("edges", [])
        aliases_raw = data.get("aliases", [])

        nodes: list[GraphNode] = []
        for node in nodes_raw:
            name = str(node.get("name", "")).strip()
            node_type = _normalize_node_type(node.get("type", "concept"))
            if not name:
                continue
            nodes.append(
                GraphNode(
                    name=name,
                    node_type=node_type,
                    properties=dict(node.get("properties", {})),
                )
            )

        edges: list[GraphEdge] = []
        for edge in edges_raw:
            source = str(edge.get("source", "")).strip()
            relation = _normalize_relation_type(edge.get("type", "related_to"))
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue
            edges.append(
                GraphEdge(
                    source=source,
                    relation=relation,
                    target=target,
                    properties=dict(edge.get("properties", {})),
                )
            )

        aliases: list[GraphAlias] = []
        for alias in aliases_raw:
            alias_name = str(alias.get("alias", "")).strip()
            canonical_name = str(alias.get("canonical", "")).strip()
            if not alias_name or not canonical_name:
                continue
            aliases.append(
                GraphAlias(
                    alias=alias_name,
                    canonical=canonical_name,
                    properties=dict(alias.get("properties", {})),
                )
            )

        edges = _deduplicate_edges(edges)

        if nodes or edges or aliases:
            return {
                "nodes": [n.to_dict() for n in nodes],
                "edges": [e.to_dict() for e in edges],
                "aliases": [a.to_dict() for a in aliases],
            }
    except Exception as exc:
        log.warning("Graph extraction failed for %s: %s", page_url or title, exc)

    nodes, edges, aliases = _fallback_graph(
        title=title,
        page_url=page_url,
        text=text,
        outgoing_links=outgoing_links or [],
        chunk_index=chunk_index,
    )
    return {
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
        "aliases": [a.to_dict() for a in aliases],
    }


def insert_graph_data(
    graph: dict[str, list[dict[str, Any]]],
    source_url: str | None = None,
    source_title: str | None = None,
    chunk_id: int | None = None,
) -> None:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    aliases = graph.get("aliases", [])

    def cypher_escape(value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace("'", "\\'")

    with get_ladybug_conn() as conn:
        for node in nodes:
            properties = dict(node.get("properties", {}))
            if source_url:
                properties.setdefault("source_url", source_url)
            if source_title:
                properties.setdefault("source_title", source_title)
            if chunk_id is not None:
                properties.setdefault("chunk_id", chunk_id)

            label = _sanitize_label(_normalize_node_type(node.get("type", "concept")))
            name = cypher_escape(node.get("name", ""))
            conn.execute(f"MERGE (n:{label} {{name: '{name}'}})")
            if properties:
                prop_clause = ", ".join(
                    f"{key}: '{cypher_escape(value)}'" for key, value in properties.items()
                )
                conn.execute(
                    f"MATCH (n:{label} {{name: '{name}'}}) SET n += {{{prop_clause}}}"
                )

        for edge in edges:
            properties = dict(edge.get("properties", {}))
            if source_url:
                properties.setdefault("source_url", source_url)
            if source_title:
                properties.setdefault("source_title", source_title)
            if chunk_id is not None:
                properties.setdefault("chunk_id", chunk_id)

            relation = _sanitize_label(_normalize_relation_type(edge.get("type", "related_to")))
            source = cypher_escape(edge.get("source", ""))
            target = cypher_escape(edge.get("target", ""))
            conn.execute(
                f"MATCH (a {{name: '{source}'}}), (b {{name: '{target}'}}) MERGE (a)-[:{relation}]->(b)"
            )
            if properties:
                prop_clause = ", ".join(
                    f"{key}: '{cypher_escape(value)}'" for key, value in properties.items()
                )
                conn.execute(
                    f"MATCH (a {{name: '{source}'}})-[r:{relation}]->(b {{name: '{target}'}}) SET r += {{{prop_clause}}}"
                )

        for alias in aliases:
            alias_name = str(alias.get("alias", "")).strip()
            canonical_name = str(alias.get("canonical", "")).strip()
            if not alias_name or not canonical_name:
                continue
            conn.execute(f"MERGE (a:concept {{name: '{cypher_escape(alias_name)}'}})")
            conn.execute(f"MERGE (c:concept {{name: '{cypher_escape(canonical_name)}'}})")
            conn.execute(
                f"MATCH (a:concept {{name: '{cypher_escape(alias_name)}'}}), (c:concept {{name: '{cypher_escape(canonical_name)}'}}) MERGE (a)-[:alias_of]->(c)"
            )
            conn.execute(
                f"MATCH (a:concept {{name: '{cypher_escape(alias_name)}'}}), (c:concept {{name: '{cypher_escape(canonical_name)}'}}) MERGE (a)-[:same_as]->(c)"
            )

        conn.commit()
