from __future__ import annotations

import logging

import click

from src.db.schema import init_all_schemas
from src.db.queues import (
    claim_pending_ingestion,
    mark_ingestion_completed,
    mark_ingestion_error,
    claim_pending_eval,
    update_eval_result,
    claim_pending_grade,
    update_grader_scores,
)
from src.ingestion.contextual_pipeline import ingest_row
from src.evaluation.pipelines import run_pipeline_a, run_pipeline_b, run_pipeline_c
from src.grading.grader import grade_answer, compute_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--phase",
    type=click.Choice(["ingest", "eval", "grade", "crawl", "stage", "extract"]),
    required=True,
)
@click.option("--batch-size", type=int, default=1)
@click.option("--model-path", type=click.Path(exists=True), default=None)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--limit", type=int, default=None)
@click.option("--start-path", default="/wiki/Main_Page")
def main(
    phase: str,
    batch_size: int,
    model_path: str | None,
    dry_run: bool,
    limit: int | None,
    start_path: str,
) -> None:
    if model_path:
        from src.models.config import MODELS

        for key in MODELS:
            MODELS[key]["path"] = model_path

    init_all_schemas()
    log.info("Initialized all schemas")

    if phase == "ingest":
        run_ingestion_loop(batch_size, dry_run)
    elif phase == "eval":
        run_eval_loop(batch_size, dry_run)
    elif phase == "grade":
        run_grade_loop(batch_size, dry_run)
    elif phase == "crawl":
        from src.ingestion.crawler import crawl as run_crawl

        run_crawl(start_path=start_path, limit=limit)
    elif phase == "stage":
        from src.ingestion.staging import (
            stage_pages,
            stage_edges,
            load_staging_to_ingestion,
        )

        stage_pages()
        stage_edges()
        load_staging_to_ingestion()
    elif phase == "extract":
        run_entity_extraction_loop(batch_size, dry_run)


def run_ingestion_loop(batch_size: int, dry_run: bool) -> None:
    log.info("Starting ingestion loop")
    while True:
        rows = []
        for _ in range(max(1, batch_size)):
            row = claim_pending_ingestion()
            if row is None:
                break
            rows.append(row)

        if not rows:
            log.info("No pending ingestion jobs")
            break

        for row in rows:
            log.info(f"Processing chunk_id={row.chunk_id}")
            if dry_run:
                mark_ingestion_completed(row.chunk_id)
                continue
            try:
                ingest_row(row)
                mark_ingestion_completed(row.chunk_id)
                log.info(f"Completed chunk_id={row.chunk_id}")
            except Exception as e:
                log.error(f"Error processing chunk_id={row.chunk_id}: {e}")
                mark_ingestion_error(row.chunk_id, row.graph_extraction_attempts + 1)


def run_eval_loop(batch_size: int, dry_run: bool) -> None:
    log.info("Starting evaluation loop")
    while True:
        row = claim_pending_eval()
        if row is None:
            log.info("No pending evaluation jobs")
            break
        log.info(f"Processing eval_id={row.eval_id}")
        if dry_run:
            for pipeline in ["a", "b", "c"]:
                update_eval_result(row.eval_id, pipeline, "dry-run result")
            continue
        try:
            result_a = run_pipeline_a(row.query)
            update_eval_result(row.eval_id, "a", result_a)
            result_b = run_pipeline_b(row.query)
            update_eval_result(row.eval_id, "b", result_b)
            result_c = run_pipeline_c(row.query)
            update_eval_result(row.eval_id, "c", result_c)
            log.info(f"Completed eval_id={row.eval_id}")
        except Exception as e:
            log.error(f"Error processing eval_id={row.eval_id}: {e}")


def run_grade_loop(batch_size: int, dry_run: bool) -> None:
    log.info("Starting grading loop")
    while True:
        row = claim_pending_grade()
        if row is None:
            log.info("No pending grading jobs")
            break
        log.info(f"Grading eval_id={row.eval_id}")
        if dry_run:
            update_grader_scores(row.eval_id, 0.5, 0.5, 0.5)
            continue
        try:
            score_a = None
            score_b = None
            score_c = None
            if row.pipeline_a_result:
                result_a = grade_answer(
                    row.query, row.ground_truth, row.pipeline_a_result, ""
                )
                score_a = compute_score(result_a)
            if row.pipeline_b_result:
                result_b = grade_answer(
                    row.query, row.ground_truth, row.pipeline_b_result, ""
                )
                score_b = compute_score(result_b)
            if row.pipeline_c_result:
                result_c = grade_answer(
                    row.query, row.ground_truth, row.pipeline_c_result, ""
                )
                score_c = compute_score(result_c)
            update_grader_scores(row.eval_id, score_a, score_b, score_c)
            log.info(f"Graded eval_id={row.eval_id}")
        except Exception as e:
            log.error(f"Error grading eval_id={row.eval_id}: {e}")


def run_entity_extraction_loop(batch_size: int, dry_run: bool) -> None:
    log.info("Legacy extract phase now routes through contextual ingestion")
    run_ingestion_loop(batch_size, dry_run)


def mark_page_extracted(page_id: int) -> None:
    from src.db.connection import get_staging_conn

    with get_staging_conn() as conn:
        conn.execute(
            "UPDATE staging_pages SET status = 'EXTRACTED' WHERE page_id = ?",
            (page_id,),
        )
        conn.commit()


if __name__ == "__main__":
    main()
