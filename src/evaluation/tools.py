from __future__ import annotations

from langchain_core.tools import tool
from src.ingestion.contextual_pipeline import get_entity_relationships as _relationships
from src.ingestion.contextual_pipeline import search_knowledge as _search


@tool
def search_knowledge(keywords: str) -> str:
    """Search contextualized Coppermind passages and return citation strings."""
    return _search(keywords)


@tool
def get_entity_relationships(entity_name: str) -> str:
    """Query Ladybug graph relationships and return citation strings."""
    return _relationships(entity_name)
