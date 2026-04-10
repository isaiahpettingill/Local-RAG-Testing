from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.models.chat_llamacpp import ChatLlamaCpp

log = logging.getLogger(__name__)

UNIVERSE_HINTS = {
    "cosmere": "Cosmere",
    "cytoverse": "Cytoverse",
    "reckoners": "Reckoners",
    "wheel of time": "Wheel of Time",
    "dark one": "Dark One",
    "standalone": "Standalone",
}

ONTOLOGY_TYPES = (
    "universe",
    "world",
    "series",
    "book",
    "story",
    "character",
    "concept",
    "organization",
    "location",
    "event",
    "artifact",
    "species",
    "power",
)


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


def _safe_title(text: str) -> str:
    return text.replace("\n", " ").strip()[:160]


def _guess_universe(title: str, text: str) -> str:
    haystack = f"{title} {text}".lower()
    for needle, universe in UNIVERSE_HINTS.items():
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
    if re.search(
        r"\b(character|person|people|born|father|mother|son|daughter)\b", text_l
    ):
        return "character"
    if re.search(
        r"\b(world|planet|city|kingdom|continent|moon|planetary|realm)\b", text_l
    ):
        return "world"
    if re.search(
        r"\b(concept|magic|power|ability|order|organization|species)\b", text_l
    ):
        return "concept"
    return "concept"


def _build_ontology_prompt(
    text: str,
    title: str,
    page_url: str,
    outgoing_links: list[str],
    chunk_index: int,
) -> str:
    links_str = "\n".join(outgoing_links[:20]) or "(none)"
    universe = _guess_universe(title, text)
    kind = _infer_kind(title, text)

    return (
        "You are extracting a semantic knowledge graph from Coppermind pages about Brandon Sanderson's works.\n"
        "Use this ontology: Universe > World > Series > Book/Story > Character/Concept/Organization/Location/Event/Artifact/Species/Power.\n"
        "Prefer high-precision nodes over exhaustive noise. Merge aliases when they clearly refer to the same thing.\n\n"
        f"Page title: {title}\n"
        f"Page URL: {page_url}\n"
        f"Chunk index: {chunk_index}\n"
        f"Likely universe: {universe}\n"
        f"Likely page kind: {kind}\n"
        f"Outgoing links:\n{links_str}\n\n"
        "Return JSON only with keys: nodes, edges.\n"
        "Each node must have: name, type, properties.\n"
        "Each edge must have: source, type, target, properties.\n"
        "Useful edge types include: belongs_to_universe, contains_world, part_of_series, appears_in, mentions, related_to, adapted_from, located_in, member_of, has_power, uses, opposes, succeeds, precedes.\n\n"
        f"Text:\n{text[:6000]}\n"
    )


def _load_json_payload(result: str) -> dict[str, Any]:
    data = json.loads(result)
    return data if isinstance(data, dict) else {}


def _fallback_graph(
    title: str, page_url: str, text: str, outgoing_links: list[str], chunk_index: int
) -> tuple[list[GraphNode], list[GraphEdge]]:
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

    edges = []
    if universe != "Unknown":
        edges.append(
            GraphEdge(
                source=_safe_title(title) or page_url,
                relation="belongs_to_universe",
                target=universe,
                properties={"source_url": page_url, "chunk_index": chunk_index},
            )
        )

    for link in outgoing_links[:5]:
        edges.append(
            GraphEdge(
                source=_safe_title(title) or page_url,
                relation="mentions",
                target=link,
                properties={"source_url": page_url, "chunk_index": chunk_index},
            )
        )

    return nodes, edges


def extract_graph(
    text: str,
    title: str = "",
    page_url: str = "",
    outgoing_links: list[str] | None = None,
    chunk_index: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    extractor = ChatLlamaCpp("graph_extractor")
    prompt = _build_ontology_prompt(
        text=text,
        title=title,
        page_url=page_url,
        outgoing_links=outgoing_links or [],
        chunk_index=chunk_index,
    )

    try:
        result = extractor.completion(prompt)
        data = _load_json_payload(result)
        nodes_raw = data.get("nodes", [])
        edges_raw = data.get("edges", [])

        nodes = []
        for node in nodes_raw:
            name = str(node.get("name", "")).strip()
            node_type = str(node.get("type", "concept")).strip() or "concept"
            if not name:
                continue
            if node_type not in ONTOLOGY_TYPES:
                node_type = "concept"
            nodes.append(
                GraphNode(
                    name=name,
                    node_type=node_type,
                    properties=dict(node.get("properties", {})),
                )
            )

        edges = []
        for edge in edges_raw:
            source = str(edge.get("source", "")).strip()
            relation = str(edge.get("type", "related_to")).strip() or "related_to"
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

        if nodes or edges:
            return {
                "nodes": [n.to_dict() for n in nodes],
                "edges": [e.to_dict() for e in edges],
            }
    except Exception as exc:
        log.warning("Graph extraction failed for %s: %s", page_url or title, exc)

    nodes, edges = _fallback_graph(
        title=title,
        page_url=page_url,
        text=text,
        outgoing_links=outgoing_links or [],
        chunk_index=chunk_index,
    )
    return {
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
    }


def insert_graph_data(
    graph: dict[str, list[dict[str, Any]]],
    source_url: str | None = None,
    source_title: str | None = None,
    chunk_id: int | None = None,
) -> None:
    client = ChatLlamaCpp("graph_extractor")
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    def cypher_escape(value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace("'", "\\'")

    for node in nodes:
        properties = dict(node.get("properties", {}))
        if source_url:
            properties.setdefault("source_url", source_url)
        if source_title:
            properties.setdefault("source_title", source_title)
        if chunk_id is not None:
            properties.setdefault("chunk_id", chunk_id)

        props = ", ".join(
            f"{key}: '{cypher_escape(value)}'" for key, value in properties.items()
        )
        props_clause = f" {{{props}}}" if props else ""
        cypher = (
            f"MERGE (n:{node.get('type', 'concept')} {{name: '{cypher_escape(node.get('name', ''))}'}})"
            f" SET n += {{{props}}}"
            if props
            else f"MERGE (n:{node.get('type', 'concept')} {{name: '{cypher_escape(node.get('name', ''))}'}})"
        )
        client.completion(cypher)

    for edge in edges:
        properties = dict(edge.get("properties", {}))
        if source_url:
            properties.setdefault("source_url", source_url)
        if source_title:
            properties.setdefault("source_title", source_title)
        if chunk_id is not None:
            properties.setdefault("chunk_id", chunk_id)

        props = ", ".join(
            f"{key}: '{cypher_escape(value)}'" for key, value in properties.items()
        )
        props_clause = f" {{{props}}}" if props else ""
        cypher = (
            f"MATCH (a {{name: '{cypher_escape(edge.get('source', ''))}'}}), "
            f"(b {{name: '{cypher_escape(edge.get('target', ''))}'}}) "
            f"MERGE (a)-[r:{edge.get('type', 'related_to')}]->(b)"
        )
        if props:
            cypher += f" SET r += {{{props}}}"
        client.completion(cypher)
