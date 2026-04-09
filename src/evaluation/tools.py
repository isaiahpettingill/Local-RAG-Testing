from __future__ import annotations

from langchain_core.tools import tool

from src.ingestion.vector_store import (
    execute_fts,
    execute_vector,
    reciprocal_rank_fusion,
)
from src.db.connection import get_ladybug_conn
from src.models.config import RECIPROCAL_RANK_FUSION_LIMIT


@tool
def search_knowledge(keywords: str) -> str:
    """Search the knowledge base for general info. Combines Vector + BM25."""
    from src.ingestion.embed import embed_query

    query_vector = embed_query(keywords)
    fts_results = execute_fts(keywords, limit=5)
    vec_results = execute_vector(query_vector, limit=5)
    hybrid_results = reciprocal_rank_fusion(
        fts_results, vec_results, limit=RECIPROCAL_RANK_FUSION_LIMIT
    )
    return "\n".join(
        [f"[Source: {r['id']}] {r['text']} ({r['url']})" for r in hybrid_results]
    )


@tool
def get_entity_relationships(entity_name: str) -> str:
    """Find everything a specific entity is directly related to."""
    client = get_ladybug_conn()
    query = f"MATCH (a {{name: '{entity_name}'}})-[r]-(b) RETURN a.name, TYPE(r), b.name, r.source_id LIMIT 5"
    result = client.execute(query).get_as_df()
    return "\n".join(
        [
            f"[Source: {r['r.source_id']}] {r['a.name']} {r['TYPE(r)']} {r['b.name']}"
            for _, r in result.iterrows()
        ]
    )
