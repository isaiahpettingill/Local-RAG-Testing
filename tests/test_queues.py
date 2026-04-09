from __future__ import annotations

import pytest
from src.db.queues import IngestionRow, EvalRow


def test_ingestion_row_namedtuple():
    row = IngestionRow(
        chunk_id=1,
        raw_text="test",
        url="https://coppermind.net/wiki/Test",
        status="PENDING",
        graph_extraction_attempts=0,
    )
    assert row.chunk_id == 1
    assert row.raw_text == "test"
    assert row.url == "https://coppermind.net/wiki/Test"
    assert row.status == "PENDING"
    assert row.graph_extraction_attempts == 0


def test_eval_row_namedtuple():
    row = EvalRow(
        eval_id=1,
        query="test query",
        ground_truth="test answer",
        pipeline_a_status="PENDING",
        pipeline_b_status="PENDING",
        pipeline_c_status="PENDING",
        pipeline_a_result=None,
        pipeline_b_result=None,
        pipeline_c_result=None,
        grader_status="PENDING",
        grader_score_a=None,
        grader_score_b=None,
        grader_score_c=None,
    )
    assert row.eval_id == 1
    assert row.query == "test query"
    assert row.ground_truth == "test answer"


def test_status_transitions():
    valid_transitions = {
        "PENDING": ["PROCESSING"],
        "PROCESSING": ["COMPLETED", "ERROR"],
        "ERROR": ["PENDING"],
    }
    assert valid_transitions["PENDING"] == ["PROCESSING"]
    assert valid_transitions["PROCESSING"] == ["COMPLETED", "ERROR"]
    assert valid_transitions["ERROR"] == ["PENDING"]
