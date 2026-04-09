from __future__ import annotations

import pytest
from src.ingestion.entity_extract import ExtractedEntity, EntityRelation


def test_extracted_entity_to_dict():
    entity = ExtractedEntity(
        name="Vin",
        entity_type="person",
        properties={"book": "Mistborn", "abilities": "Allomancy"},
    )
    d = entity.to_dict()
    assert d["name"] == "Vin"
    assert d["entity_type"] == "person"
    assert d["properties"]["book"] == "Mistborn"
    assert d["properties"]["abilities"] == "Allomancy"


def test_entity_relation_to_dict():
    relation = EntityRelation(
        from_entity="Vin",
        to_entity="Kelsier",
        relation_type="related_to",
        source_page="https://coppermind.net/wiki/Mistborn",
    )
    d = relation.to_dict()
    assert d["from_entity"] == "Vin"
    assert d["to_entity"] == "Kelsier"
    assert d["relation_type"] == "related_to"
    assert d["source_page"] == "https://coppermind.net/wiki/Mistborn"
