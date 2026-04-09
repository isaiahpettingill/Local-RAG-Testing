from __future__ import annotations

from langchain_core.tools import StructuredTool


def test_search_knowledge_output_format():
    from src.evaluation.tools import search_knowledge

    assert isinstance(search_knowledge, StructuredTool)
    assert search_knowledge.name == "search_knowledge"


def test_get_entity_relationships_output_format():
    from src.evaluation.tools import get_entity_relationships

    assert isinstance(get_entity_relationships, StructuredTool)
    assert get_entity_relationships.name == "get_entity_relationships"
