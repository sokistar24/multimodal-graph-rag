"""Binary LLM judges that justify each verdict in one sentence.

The text judge grades accuracy and faithfulness on text evidence; the
vision judge grades faithfulness on figure questions with the crop attached,
because a text-only model cannot check an answer against an image it cannot
see. Each judge writes a one-sentence justification and then the verdict on a
line of its own; parsing stays binary, and the full reply is kept with every
verdict so it can be audited later.

Judges are a tier above the generators they grade, but not always from a
different family: GPT-6.1 Sol grades gpt4o-mini, and DeepSeek V4 Pro grades
the DeepSeek-authored question sets. Both overlaps are stated limitations.

These judges are NOT calibrated against human labels collected for this study.
Where the reference answer is a short span, prefer the reference-based metrics
in :mod:`..evaluation.answer_match`: exact match and token F1 are computed
against answers the benchmark's own annotators wrote, so they carry human
judgement without a judge model in the loop. Report a judge verdict as the
primary outcome only where no reference-based metric applies, and say so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..clients import GenResult, ModelClient

VERDICT_INSTRUCTION = (
    "First write one sentence explaining your decision. Then, on a new final "
    "line, write only the digit {one} or {zero}."
)
ANSWER_PROMPT = (
    "You are grading a question-answering system. Decide whether the GENERATED "
    "answer is factually correct, given the EXPECTED answer. Minor wording "
    "differences are fine.\n\n"
    "QUESTION: {question}\nEXPECTED: {expected}\nGENERATED: {generated}\n\n"
    + VERDICT_INSTRUCTION.format(one="1 (correct)", zero="0 (incorrect)")
)
TEXT_FAITHFULNESS_PROMPT = (
    "Check whether an answer is FAITHFUL to the context (every claim supported, "
    "nothing invented).\n\n"
    "CONTEXT:\n{context}\n\nANSWER:\n{generated}\n\n"
    + VERDICT_INSTRUCTION.format(one="1 (faithful)", zero="0 (not faithful)")
)
VISION_FAITHFULNESS_PROMPT = (
    "Check whether the ANSWER is faithful to the evidence: every claim "
    "supported by the attached image and/or the text context, nothing "
    "invented.\n\n"
    "TEXT CONTEXT:\n{context}\n\nANSWER:\n{generated}\n\n"
    + VERDICT_INSTRUCTION.format(one="1 (faithful)", zero="0 (not faithful)")
)


@dataclass(frozen=True)
class Verdict:
    value: int
    call: GenResult

    @property
    def justification(self) -> str:
        """Everything the judge wrote before the verdict line."""
        lines = [line for line in self.call.text.splitlines() if line.strip()]
        return "\n".join(lines[:-1]).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "judge": self.call.model,
            "justification": self.justification,
            "response": self.call.text,
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
