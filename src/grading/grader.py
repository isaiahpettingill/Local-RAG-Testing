from __future__ import annotations

import json


class GradeResult:
    def __init__(
        self, accurate: bool, correct_synthesis: bool, no_hallucination: bool
    ) -> None:
        self.accurate = accurate
        self.correct_synthesis = correct_synthesis
        self.no_hallucination = no_hallucination

    def to_dict(self) -> dict[str, bool]:
        return {
            "accurate": self.accurate,
            "correct_synthesis": self.correct_synthesis,
            "no_hallucination": self.no_hallucination,
        }


def grade_answer(
    query: str, ground_truth: str, answer: str, context: str
) -> GradeResult:
    from src.models.chat_llamacpp import ChatLlamaCpp

    grader = ChatLlamaCpp("grader")
    prompt = (
        "You are grading a RAG system answer. Evaluate the following answer against the ground truth.\n"
        f"Query: {query}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Answer: {answer}\n"
        f"Context: {context}\n\n"
        "Return JSON with fields: accurate (bool), correct_synthesis (bool), no_hallucination (bool)."
    )
    result = grader.completion(prompt)
    try:
        data = json.loads(result)
        return GradeResult(
            accurate=bool(data.get("accurate", False)),
            correct_synthesis=bool(data.get("correct_synthesis", False)),
            no_hallucination=bool(data.get("no_hallucination", False)),
        )
    except json.JSONDecodeError:
        return GradeResult(
            accurate=False, correct_synthesis=False, no_hallucination=False
        )


def compute_score(result: GradeResult) -> float:
    return (
        sum([result.accurate, result.correct_synthesis, result.no_hallucination]) / 3.0
    )
