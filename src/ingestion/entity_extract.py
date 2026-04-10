from __future__ import annotations

import json
import logging
from typing import Any

from src.models.chat_llamacpp import ChatLlamaCpp

log = logging.getLogger(__name__)

ENTITY_TYPES = (
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

ENTITY_TYPE_ALIASES = {
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

RELATION_ALIASES = {
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
    "wields": "wields",
    "connected_to": "related_to",
    "references": "mentions",
    "alias": "alias_of",
    "aka": "alias_of",
    "same": "same_as",
}


class ExtractedEntity:
    def __init__(
        self,
        name: str,
        entity_type: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.entity_type = entity_type
        self.properties = properties or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
        }


class EntityRelation:
    def __init__(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        source_page: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relation_type = relation_type
        self.source_page = source_page
        self.properties = properties or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "source_page": self.source_page,
            "properties": self.properties,
        }


def _normalize_entity_type(value: str | None) -> str:
    if not value:
        return "concept"
    normalized = value.strip().lower().replace(" ", "_")
    return ENTITY_TYPE_ALIASES.get(normalized, normalized if normalized in ENTITY_TYPES else "concept")


def _normalize_relation_type(value: str | None) -> str:
    if not value:
        return "related_to"
    normalized = value.strip().lower().replace(" ", "_")
    return RELATION_ALIASES.get(normalized, normalized if normalized in RELATION_TYPES else "related_to")


def _build_entity_extraction_prompt(text: str, title: str, page_url: str) -> str:
    return (
        "You are extracting a moderate semantic knowledge graph for Brandon Sanderson fiction.\n"
        "Return only the most useful canonical entities for retrieval.\n"
        "Use these entity types only: universe, series, book, world, character, faction, magic_system, artifact, event, concept.\n"
        "Prefer canonical names over aliases. If a character or concept has an alias, include it in the aliases list on the canonical entity.\n"
        "Do not output Cypher or prose.\n\n"
        f"Page title: {title}\n"
        f"Page URL: {page_url}\n\n"
        "Return JSON only with this shape:\n"
        '{"entities": [{"name": "...", "type": "character", "properties": {"aliases": ["..."], "summary": "...", "confidence": 0.0}}]}\n\n'
        f"Text:\n{text[:5000]}\n"
    )


def _build_relation_extraction_prompt(
    text: str,
    entities: list[ExtractedEntity],
    title: str,
    page_url: str,
    outgoing_links: list[str],
    existing_relationships: str = "",
) -> str:
    entity_names = [e.name for e in entities]
    links_str = "\n".join(outgoing_links[:10]) or "(none)"
    existing = existing_relationships.strip() or "(none)"

    return (
        "You are extracting relationships for a Sanderson semantic graph.\n"
        "Use only these relation types: part_of_universe, part_of_series, appears_in_book, set_on_world, member_of, leads, ally_of, enemy_of, mentor_of, student_of, family_of, uses_magic_system, wields, associated_with_artifact, associated_with_event, originates_from_world, located_in, same_as, alias_of, mentions, related_to.\n"
        "Avoid duplicates when the same relationship already exists in the graph context below.\n"
        "If a character has an alias, output alias_of from the alias to the canonical name.\n"
        "Return only JSON.\n\n"
        f"Page title: {title}\n"
        f"Page URL: {page_url}\n"
        f"Entities: {entity_names}\n"
        f"Outgoing links:\n{links_str}\n\n"
        f"Existing similar relationships already in the graph:\n{existing}\n\n"
        "Return JSON in this shape:\n"
        '{"relations": [{"from": "...", "to": "...", "type": "alias_of", "source": "...", "properties": {"confidence": 0.0}}]}\n\n'
        f"Text:\n{text[:5000]}\n"
    )


def _load_entities_payload(response: str) -> list[dict[str, Any]]:
    data = json.loads(response)
    if not isinstance(data, dict):
        return []
    entities_data = data.get("entities", [])
    return entities_data if isinstance(entities_data, list) else []


def _load_relations_payload(response: str) -> list[dict[str, Any]]:
    data = json.loads(response)
    if not isinstance(data, dict):
        return []
    relations_data = data.get("relations", [])
    return relations_data if isinstance(relations_data, list) else []


def extract_entities(text: str, title: str = "", page_url: str = "") -> list[ExtractedEntity]:
    model = ChatLlamaCpp("graph_extractor")
    prompt = _build_entity_extraction_prompt(text, title, page_url)

    try:
        response = model.completion(prompt)
        entities_data = _load_entities_payload(response)

        entities = []
        for ent in entities_data:
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=_normalize_entity_type(ent.get("type")),
                    properties=dict(ent.get("properties", {})),
                )
            )
        return entities
    except json.JSONDecodeError:
        log.warning("Failed to parse entity extraction response")
        return []
    except Exception as e:
        log.error("Entity extraction failed: %s", e)
        return []


def extract_relations(
    text: str,
    entities: list[ExtractedEntity],
    title: str = "",
    page_url: str = "",
    outgoing_links: list[str] | None = None,
    existing_relationships: str = "",
) -> list[EntityRelation]:
    if not entities:
        return []

    model = ChatLlamaCpp("graph_extractor")
    prompt = _build_relation_extraction_prompt(
        text=text,
        entities=entities,
        title=title,
        page_url=page_url,
        outgoing_links=outgoing_links or [],
        existing_relationships=existing_relationships,
    )

    try:
        response = model.completion(prompt)
        relations_data = _load_relations_payload(response)

        relations = []
        for rel in relations_data:
            from_entity = str(rel.get("from", "")).strip()
            to_entity = str(rel.get("to", "")).strip()
            if not from_entity or not to_entity:
                continue
            relations.append(
                EntityRelation(
                    from_entity=from_entity,
                    to_entity=to_entity,
                    relation_type=_normalize_relation_type(rel.get("type")),
                    source_page=str(rel.get("source", page_url or title)),
                    properties=dict(rel.get("properties", {})),
                )
            )
        return relations
    except json.JSONDecodeError:
        log.warning("Failed to parse relation extraction response")
        return []
    except Exception as e:
        log.error("Relation extraction failed: %s", e)
        return []


def extract_entities_and_relations(
    text: str,
    title: str = "",
    page_url: str = "",
    outgoing_links: list[str] | None = None,
    existing_relationships: str = "",
) -> tuple[list[ExtractedEntity], list[EntityRelation]]:
    entities = extract_entities(text, title=title, page_url=page_url)
    relations = extract_relations(
        text,
        entities,
        title=title,
        page_url=page_url,
        outgoing_links=outgoing_links,
        existing_relationships=existing_relationships,
    )
    return entities, relations
