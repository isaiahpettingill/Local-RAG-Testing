from __future__ import annotations


class _Cursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _QueueConn:
    def __init__(self):
        self.executed: list[tuple[str, tuple | None]] = []

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if "FROM ingestion_queue" in query:
            return _Cursor(
                {
                    "chunk_id": 9,
                    "staging_page_id": 42,
                    "status": "PENDING",
                    "graph_extraction_attempts": 0,
                }
            )
        return _Cursor(None)

    def commit(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _StagingConn:
    def execute(self, query, params=None):
        assert params == (42,)
        return _Cursor(
            {
                "text": "row text",
                "url": "https://example.com/page",
                "title": "Page Title",
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_claim_pending_ingestion_hydrates_from_staging(monkeypatch):
    import src.db.queues as queues

    queue_conn = _QueueConn()
    monkeypatch.setattr(queues, "get_ingestion_queue_conn", lambda: queue_conn)

    import src.db.connection as connection

    monkeypatch.setattr(connection, "get_staging_conn", lambda: _StagingConn())

    row = queues.claim_pending_ingestion()

    assert row is not None
    assert row.staging_page_id == 42
    assert row.raw_text == "row text"
    assert row.url == "https://example.com/page"
    assert row.source_title == "Page Title"
    assert any("UPDATE ingestion_queue SET status = 'PROCESSING'" in q for q, _ in queue_conn.executed)
