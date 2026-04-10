from __future__ import annotations

from typing import NamedTuple

from src.db.connection import get_ingestion_queue_conn, get_eval_queue_conn


class IngestionRow(NamedTuple):
    chunk_id: int
    raw_text: str
    url: str | None
    source_title: str
    status: str
    graph_extraction_attempts: int


class EvalRow(NamedTuple):
    eval_id: int
    query: str
    ground_truth: str
    pipeline_a_status: str
    pipeline_b_status: str
    pipeline_c_status: str
    pipeline_a_result: str | None
    pipeline_b_result: str | None
    pipeline_c_result: str | None
    grader_status: str
    grader_score_a: float | None
    grader_score_b: float | None
    grader_score_c: float | None


def claim_pending_ingestion() -> IngestionRow | None:
    with get_ingestion_queue_conn() as conn:
        row = conn.execute(
            "SELECT chunk_id, raw_text, url, source_title, status, graph_extraction_attempts "
            "FROM ingestion_queue WHERE status = 'PENDING' ORDER BY chunk_id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE ingestion_queue SET status = 'PROCESSING', updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?",
            (row["chunk_id"],),
        )
        conn.commit()
        return IngestionRow(**row)


def mark_ingestion_completed(chunk_id: int) -> None:
    with get_ingestion_queue_conn() as conn:
        conn.execute(
            "UPDATE ingestion_queue SET status = 'COMPLETED', updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?",
            (chunk_id,),
        )
        conn.commit()


def mark_ingestion_error(chunk_id: int, attempts: int) -> None:
    new_status = "ERROR" if attempts >= 3 else "PENDING"
    with get_ingestion_queue_conn() as conn:
        conn.execute(
            "UPDATE ingestion_queue SET status = ?, graph_extraction_attempts = ?, updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?",
            (new_status, attempts, chunk_id),
        )
        conn.commit()


def claim_pending_eval() -> EvalRow | None:
    with get_eval_queue_conn() as conn:
        row = conn.execute(
            "SELECT eval_id, query, ground_truth, pipeline_a_status, pipeline_b_status, pipeline_c_status, "
            "pipeline_a_result, pipeline_b_result, pipeline_c_result, grader_status, "
            "grader_score_a, grader_score_b, grader_score_c "
            "FROM evaluation_queue WHERE pipeline_c_status = 'PENDING' ORDER BY eval_id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return EvalRow(**row)


def update_eval_result(eval_id: int, pipeline: str, result: str) -> None:
    col = f"pipeline_{pipeline}_result"
    status_col = f"pipeline_{pipeline}_status"
    with get_eval_queue_conn() as conn:
        conn.execute(
            f"UPDATE evaluation_queue SET {col} = ?, {status_col} = 'COMPLETED', updated_at = CURRENT_TIMESTAMP WHERE eval_id = ?",
            (result, eval_id),
        )
        conn.commit()


def claim_pending_grade() -> EvalRow | None:
    with get_eval_queue_conn() as conn:
        row = conn.execute(
            "SELECT eval_id, query, ground_truth, pipeline_a_status, pipeline_b_status, pipeline_c_status, "
            "pipeline_a_result, pipeline_b_result, pipeline_c_result, grader_status, "
            "grader_score_a, grader_score_b, grader_score_c "
            "FROM evaluation_queue WHERE grader_status = 'PENDING' AND pipeline_c_status = 'COMPLETED' ORDER BY eval_id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE evaluation_queue SET grader_status = 'PROCESSING', updated_at = CURRENT_TIMESTAMP WHERE eval_id = ?",
            (row["eval_id"],),
        )
        conn.commit()
        return EvalRow(**row)


def update_grader_scores(
    eval_id: int, score_a: float | None, score_b: float | None, score_c: float | None
) -> None:
    with get_eval_queue_conn() as conn:
        conn.execute(
            "UPDATE evaluation_queue SET grader_status = 'COMPLETED', grader_score_a = ?, grader_score_b = ?, grader_score_c = ?, updated_at = CURRENT_TIMESTAMP WHERE eval_id = ?",
            (score_a, score_b, score_c, eval_id),
        )
        conn.commit()
