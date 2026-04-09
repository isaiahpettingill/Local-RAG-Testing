from __future__ import annotations

import mo


def get_queue_counts(conn) -> dict[str, int]:
    cursor = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM ingestion_queue GROUP BY status"
    )
    counts = {"PENDING": 0, "PROCESSING": 0, "COMPLETED": 0, "ERROR": 0}
    for row in cursor.fetchall():
        counts[row["status"]] = row["cnt"]
    return counts


def get_eval_counts(conn) -> dict[str, int]:
    cursor = conn.execute(
        "SELECT grader_status, COUNT(*) as cnt FROM evaluation_queue GROUP BY grader_status"
    )
    counts = {"PENDING": 0, "PROCESSING": 0, "COMPLETED": 0, "ERROR": 0}
    for row in cursor.fetchall():
        counts[row["grader_status"]] = row["cnt"]
    return counts


@mo.page(run_on_start=True)
def dashboard():
    mo.md("# Local RAG Dashboard")

    mo.ui.refresh(intervals=5)

    ingestion_counts = {"PENDING": 0, "PROCESSING": 0, "COMPLETED": 0, "ERROR": 0}
    eval_counts = {"PENDING": 0, "PROCESSING": 0, "COMPLETED": 0, "ERROR": 0}

    mo.md(f"### Ingestion Queue: {ingestion_counts}")
    mo.md(f"### Evaluation Queue: {eval_counts}")
