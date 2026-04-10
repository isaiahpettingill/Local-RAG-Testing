from __future__ import annotations


def test_insert_chunk_persists_context_and_raw_text(monkeypatch):
    import src.ingestion.vector_store as vector_store

    class DummyCursor:
        def __init__(self, lastrowid=11):
            self.lastrowid = lastrowid

        def fetchone(self):
            return None

    class DummyConn:
        def __init__(self):
            self.executed = []

        def execute(self, query, params=None):
            self.executed.append((query, params))
            if query.startswith("SELECT id FROM chunks"):
                return DummyCursor()
            return DummyCursor()

        def commit(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    conn = DummyConn()
    monkeypatch.setattr(vector_store, "get_knowledgebase_conn", lambda: conn)
    monkeypatch.setattr(vector_store, "_pack_vector", lambda vector: b"packed")

    chunk_id = vector_store.insert_chunk(
        text="contextual chunk",
        vector=[0.1, 0.2],
        url="https://example.com/test",
        source_title="Test",
        chunk_index=3,
        raw_text="original chunk",
        context="chunk context",
    )

    assert chunk_id == 11
    assert any("raw_text" in query for query, _ in conn.executed)
    assert any("context" in query for query, _ in conn.executed)
