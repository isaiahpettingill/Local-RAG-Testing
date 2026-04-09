from __future__ import annotations

import json
import logging
from typing import Any

from src.models.chat_llamacpp import ChatLlamaCpp

log = logging.getLogger(__name__)


class ExtractedEntity:
    def __init__(
        self,
        name: str,
        entity_type: str | None = None,
        properties: dict[str, str] | None = None,
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
    ) -> None:
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relation_type = relation_type
        self.source_page = source_page

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "source_page": self.source_page,
        }


def _build_entity_extraction_prompt(text: str) -> str:
    return (
        "You are an entity extraction system. Given the text below, identify all named entities.\n"
        "For each entity, also determine its type (person, place, organization, object, concept, event) "
        "and extract relevant properties.\n\n"
        "Return a JSON object with an 'entities' field containing each entity's name, type, and properties.\n"
        'Example: {"entities": [{"name": "Vin", "type": "person", "properties": {"book": "Mistborn"}}]}\n\n'
        f"Text: {text[:4000]}\n\n"
        "Respond only with valid JSON."
    )


def _build_relation_extraction_prompt(
    text: str, entities: list[ExtractedEntity], page_url: str, outgoing_links: list[str]
) -> str:
    entity_names = [e.name for e in entities]
    links_str = "\n".join(outgoing_links[:10])

    return (
        "You are a relation extraction system. Given the text and known entities, extract relationships between entities.\n\n"
        "For each relationship, identify:\n"
        "- The source entity (from the entities list)\n"
        "- The target entity (can be from entities list OR implied by the outgoing links)\n"
        "- The relationship type (e.g., 'related_to', 'appears_in', 'references', 'co-founded', 'located_in')\n\n"
        f"Entities found: {entity_names}\n\n"
        f"Outgoing links in document (may indicate related entities):\n{links_str}\n\n"
        f"Source page URL: {page_url}\n\n"
        "Return a JSON object with a 'relations' field.\n"
        'Example: {"relations": [{"from": "Vin", "to": "Kelsier", "type": "related_to", "source": "https://coppermind.net/wiki/Mistborn"}]}\n\n'
        f"Text: {text[:4000]}\n\n"
        "Respond only with valid JSON."
    )


def extract_entities(text: str) -> list[ExtractedEntity]:
    model = ChatLlamaCpp("graph_extractor")
    prompt = _build_entity_extraction_prompt(text)

    try:
        response = model.completion(prompt)
        data = json.loads(response)
        entities_data = data.get("entities", [])

        entities = []
        for ent in entities_data:
            name = ent.get("name", "")
            if not name:
                continue
            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=ent.get("type"),
                    properties=ent.get("properties", {}),
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
    page_url: str,
    outgoing_links: list[str],
) -> list[EntityRelation]:
    if not entities:
        return []

    model = ChatLlamaCpp("graph_extractor")
    prompt = _build_relation_extraction_prompt(text, entities, page_url, outgoing_links)

    try:
        response = model.completion(prompt)
        data = json.loads(response)
        relations_data = data.get("relations", [])

        relations = []
        for rel in relations_data:
            from_entity = rel.get("from", "")
            to_entity = rel.get("to", "")
            if not from_entity or not to_entity:
                continue
            relations.append(
                EntityRelation(
                    from_entity=from_entity,
                    to_entity=to_entity,
                    relation_type=rel.get("type", "related_to"),
                    source_page=rel.get("source", page_url),
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
    text: str, page_url: str, outgoing_links: list[str]
) -> tuple[list[ExtractedEntity], list[EntityRelation]]:
    entities = extract_entities(text)
    relations = extract_relations(text, entities, page_url, outgoing_links)
    return entities, relations
