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


GRADING_CRITERIA = """
## Grading Rubric

The grader evaluates answers on three criteria:

1. **Accurate** (faithfulness): Does the answer correctly use facts from the provided context?
   - Answer must NOT contradict any information in the context
   - Answer must NOT introduce external facts not present in context
   - Score: true if faithful, false if contradictory or contains external facts

2. **Correct Synthesis**: Does the answer address the query comprehensively?
   - Answer must directly address what was asked
   - Answer must synthesize relevant information from context
   - Score: true if addresses query well, false if off-topic or incomplete

3. **No Hallucination**: Is every claim in the answer grounded in the context?
   - Each factual claim must be verifiable in the provided context
   - No made-up entities, dates, statistics, or statements
   - Score: true if all claims grounded, false if any ungrounded claim

Return JSON with fields: accurate (bool), correct_synthesis (bool), no_hallucination (bool).
"""


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
        f"Context: {context}\n\n" + GRADING_CRITERIA
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
