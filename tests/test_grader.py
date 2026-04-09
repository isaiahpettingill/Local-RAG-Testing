from __future__ import annotations

import pytest
from src.grading.grader import GradeResult, compute_score


def test_grade_result_to_dict():
    result = GradeResult(accurate=True, correct_synthesis=False, no_hallucination=True)
    d = result.to_dict()
    assert d["accurate"] is True
    assert d["correct_synthesis"] is False
    assert d["no_hallucination"] is True


def test_compute_score():
    result = GradeResult(accurate=True, correct_synthesis=True, no_hallucination=True)
    score = compute_score(result)
    assert score == 1.0

    result2 = GradeResult(
        accurate=False, correct_synthesis=False, no_hallucination=False
    )
    score2 = compute_score(result2)
    assert score2 == 0.0

    result3 = GradeResult(
        accurate=True, correct_synthesis=False, no_hallucination=False
    )
    score3 = compute_score(result3)
    assert score3 == pytest.approx(1 / 3)
