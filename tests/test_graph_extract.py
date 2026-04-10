from __future__ import annotations


class DummyModel:
    def __init__(self, response: str, captured: dict[str, str]):
        self.response = response
        self.captured = captured

    def completion(self, prompt):
        self.captured["prompt"] = prompt
        return self.response


def test_extract_graph_normalizes_aliases_and_dedupes(monkeypatch):
    import src.ingestion.graph_extract as graph_extract

    captured: dict[str, str] = {}
    response = (
        '{"nodes": ['
        '{"name": "Hoid", "type": "character", "properties": {"aliases": ["Wit"]}},'
        '{"name": "Roshar", "type": "world", "properties": {}}'
        '], '
        '"edges": ['
        '{"source": "Hoid", "type": "alias", "target": "Wit", "properties": {}},'
        '{"source": "Hoid", "type": "alias_of", "target": "Wit", "properties": {}},'
        '{"source": "Hoid", "type": "member", "target": "Bridge Four", "properties": {}}'
        '], '
        '"aliases": ['
        '{"alias": "Wit", "canonical": "Hoid", "properties": {"confidence": 0.99}}'
        ']}'
    )

    monkeypatch.setattr(
        graph_extract.ChatLlamaCpp,
        "completion",
        lambda self, prompt: DummyModel(response, captured).completion(prompt),
    )

    result = graph_extract.extract_graph(
        "Hoid is also called Wit and has a relation to Bridge Four.",
        title="Test Title",
        page_url="https://example.com/test",
        outgoing_links=["https://example.com/bridge-four"],
        chunk_index=0,
        existing_relationships="Hoid -[alias_of]-> Wit",
    )

    edge_types = [edge["type"] for edge in result["edges"]]
    assert edge_types.count("alias_of") == 1
    assert edge_types.count("member_of") == 1
    assert any(node["type"] == "character" for node in result["nodes"])
    assert result["aliases"][0]["alias"] == "Wit"


def test_extract_graph_uses_graph_summarizer_role(monkeypatch):
    import src.ingestion.graph_extract as graph_extract

    captured: dict[str, str] = {}

    monkeypatch.setattr(
        graph_extract.ChatLlamaCpp,
        "completion",
        lambda self, prompt: DummyModel('{"nodes": [], "edges": [], "aliases": []}', captured).completion(prompt),
    )

    graph_extract.extract_graph(
        "A small text about Kaladin.",
        title="The Way of Kings",
        page_url="https://example.com/way-of-kings",
        outgoing_links=[],
        chunk_index=1,
    )

    assert "Universe > Series > Book > World > Character" in captured["prompt"]
