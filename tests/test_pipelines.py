from __future__ import annotations

import pytest


def test_pipeline_a_signature():
    from src.evaluation.pipelines import run_pipeline_a

    assert callable(run_pipeline_a)


def test_pipeline_b_signature():
    from src.evaluation.pipelines import run_pipeline_b

    assert callable(run_pipeline_b)


def test_pipeline_c_signature():
    from src.evaluation.pipelines import run_pipeline_c

    assert callable(run_pipeline_c)
