"""Binary LLM judges that sit outside every generator family.

DeepSeek grades accuracy, relevancy, and faithfulness on text evidence; the
vision judge grades faithfulness on figure questions with the crop attached,
because a text-only model cannot check an answer against an image it cannot
see. Every verdict is returned with the raw call so it can be audited against
human labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..clients import GenResult, ModelClient

ANSWER_PROMPT = (
    "You are grading a question-answering system. Decide whether the GENERATED "
    "answer is factually correct, given the EXPECTED answer. Minor wording "
    "differences are fine. Reply with ONLY the digit 1 or 0.\n\n"
    "QUESTION: {question}\nEXPECTED: {expected}\nGENERATED: {generated}\n\n"
    "Grade (1 or 0):"
)
TEXT_FAITHFULNESS_PROMPT = (
    "Check whether an answer is FAITHFUL to the context (every claim supported, "
    "nothing invented). Reply with ONLY the digit 1 or 0.\n\n"
    "CONTEXT:\n{context}\n\nANSWER:\n{generated}\n\nGrade (1 or 0):"
)
VISION_FAITHFULNESS_PROMPT = (
    "Check whether the ANSWER is faithful to the evidence: every claim "
    "supported by the attached image and/or the text context, nothing "
    "invented. Reply with ONLY the digit 1 or 0.\n\n"
    "TEXT CONTEXT:\n{context}\n\nANSWER:\n{generated}\n\nGrade (1 or 0):"
)
RELEVANCY_PROMPT = (
    "Check whether an answer is RELEVANT to the question (addresses it, "
    "regardless of correctness). Reply with ONLY the digit 1 or 0.\n\n"
    "QUESTION: {question}\nANSWER: {generated}\n\nGrade (1 or 0):"
)


@dataclass(frozen=True)
class Verdict:
    value: int
    call: GenResult

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "judge": self.call.model,
            "input_tokens": self.call.input_tokens,
            "output_tokens": self.call.output_tokens,
            "cost_usd": self.call.cost_usd,
        }


class Judges:
    def __init__(
        self,
        client: ModelClient,
        *,
        text_judge: str = "deepseek",
        vision_judge: str = "claude-haiku",
    ) -> None:
        self.client = client
        self.text_judge = text_judge
        self.vision_judge = vision_judge
        if not client.spec(vision_judge).vision:
            raise ValueError(f"vision judge {vision_judge} cannot see images")

    def answer(self, question: str, expected: str, generated: str) -> Verdict:
        value, call = self.client.judge(
            self.text_judge,
            ANSWER_PROMPT.format(
                question=question, expected=expected, generated=generated
            ),
        )
        return Verdict(value, call)

    def faithfulness(
        self, generated: str, context: str, image_paths: Sequence[Path] = ()
    ) -> Verdict:
        if image_paths:
            value, call = self.client.judge(
                self.vision_judge,
                VISION_FAITHFULNESS_PROMPT.format(context=context, generated=generated),
                images=image_paths,
            )
        else:
            value, call = self.client.judge(
                self.text_judge,
                TEXT_FAITHFULNESS_PROMPT.format(context=context, generated=generated),
            )
        return Verdict(value, call)

    def relevancy(self, question: str, generated: str) -> Verdict:
        value, call = self.client.judge(
            self.text_judge,
            RELEVANCY_PROMPT.format(question=question, generated=generated),
        )
        return Verdict(value, call)
