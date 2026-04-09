from __future__ import annotations

from typing import Any

from src.db.connection import get_ladybug_conn


def extract_graph(text: str) -> list[dict[str, Any]]:
    from src.models.chat_llamacpp import ChatLlamaCpp

    extractor = ChatLlamaCpp("graph_extractor")
    prompt = (
        "Extract entities and relationships from the following text. "
        "Return a JSON list of objects with 'source', 'type', 'target', 'source_id' fields.\n\n"
        f"Text: {text}"
    )
    result = extractor.completion(prompt)
    import json

    try:
        data = json.loads(result)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def insert_graph_data(
    triples: list[dict[str, Any]], source_url: str | None = None, chunk_id: int | None = None
) -> None:
    client = get_ladybug_conn()
    for triple in triples:
        props = {}
        if source_url:
            props["source"] = source_url
        if chunk_id:
            props["chunk_id"] = chunk_id
        
        properties_str = ", ".join([f"{k}: '{v}'" for k, v in props.items()])
        properties = f" {{ {properties_str} }}" if properties_str else ""

        client.execute(
            f"MERGE (a:Entity {{name: '{triple['source']}'}}) "
            f"MERGE (b:Entity {{name: '{triple['target']}'}}) "
            f"MERGE (a)-[r:{triple['type']}]->(b)"
            + (f" SET r += {properties}" if properties else "")
        )
