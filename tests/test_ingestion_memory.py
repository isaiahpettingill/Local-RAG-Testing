from __future__ import annotations


class _Row:
    def __init__(self, chunk_id: int):
        self.chunk_id = chunk_id
        self.graph_extraction_attempts = 0


def test_run_ingestion_loop_processes_without_batch_retention(monkeypatch):
    import cli

    queue = [_Row(1), _Row(2), _Row(3)]
    seen: list[int] = []

    def claim():
        return queue.pop(0) if queue else None

    monkeypatch.setattr(cli, "claim_pending_ingestion", claim)
    monkeypatch.setattr(cli, "ingest_row", lambda row, graph_enabled=True: seen.append(row.chunk_id))
    monkeypatch.setattr(cli, "mark_ingestion_completed", lambda chunk_id: None)
    monkeypatch.setattr(cli, "mark_ingestion_error", lambda chunk_id, attempts: None)

    cli.run_ingestion_loop(batch_size=50, dry_run=False, no_graph=True)

    assert seen == [1, 2, 3]
