from __future__ import annotations


def test_build_context_prompt_includes_metadata():
    from src.ingestion.contextual_pipeline import _build_context_prompt

    prompt = _build_context_prompt(
        document_text="alpha beta gamma",
        title="Test Title",
        url="https://example.com/test",
        chunk_text="beta gamma",
        chunk_index=1,
        total_chunks=4,
    )

    assert "Test Title" in prompt
    assert "https://example.com/test" in prompt
    assert "Chunk position: 2 of 4" in prompt
    assert "beta gamma" in prompt


def test_build_contextual_chunks_uses_generated_context(monkeypatch):
    import src.ingestion.contextual_pipeline as pipeline

    monkeypatch.setattr(pipeline, "chunk_text", lambda text, chunk_size=1000, chunk_overlap=200: ["chunk one", "chunk two"])
    monkeypatch.setattr(
        pipeline,
        "_generate_chunk_context",
        lambda document_text, title, url, chunk_text, chunk_index, total_chunks: f"context-{chunk_index}",
    )

    chunks = pipeline.build_contextual_chunks(
        "ignored document text",
        title="Title",
        url="https://example.com/test",
    )

    assert [chunk.raw_text for chunk in chunks] == ["chunk one", "chunk two"]
    assert chunks[0].context == "context-0"
    assert chunks[0].contextual_text == "context-0\n\nchunk one"
    assert chunks[1].contextual_text == "context-1\n\nchunk two"


def test_search_knowledge_fuses_results(monkeypatch):
    import src.ingestion.contextual_pipeline as pipeline

    monkeypatch.setattr(pipeline, "embed_query", lambda text: [1.0, 2.0])
    monkeypatch.setattr(
        pipeline,
        "execute_fts",
        lambda keywords, limit=5: [{"id": 1, "text": "fts result", "url": "u1"}],
    )
    monkeypatch.setattr(
        pipeline,
        "execute_vector",
        lambda query_vector, limit=5: [{"id": 2, "text": "vector result", "url": "u2"}],
    )

    output = pipeline.search_knowledge("query", limit=2)

    assert "[Source:" in output
    assert "fts result" in output or "vector result" in output
