from __future__ import annotations

import pytest


def test_search_knowledge_output_format():
    from src.evaluation.tools import search_knowledge

    assert callable(search_knowledge)


def test_get_entity_relationships_output_format():
    from src.evaluation.tools import get_entity_relationships

    assert callable(get_entity_relationships)
