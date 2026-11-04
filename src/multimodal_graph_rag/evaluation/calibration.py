"""Synthetic calibration of an accuracy judge against the references alone.

For every question the judge grades two answers it has never seen from a
generator: the question's own reference, which must be graded correct, and
the reference of another question from the same set, which must be graded
incorrect. The decoy is the next question in set order whose normalised
reference differs, so a "yes" question never receives a "yes" decoy. The two
error rates bound how often the judge rejects a correct answer and accepts a
wrong one. This is a sanity floor, not calibration against human labels.

Verdicts are appended to a JSON-lines file keyed by question and kind, so an
interrupted pass resumes without repeating calls.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..clients import ModelClient
from ..errors import ConfigurationError
from ..schemas import QuestionRecord
from .answer_match import normalise_answer
from .judges import ANSWER_PROMPT

KINDS = ("reference", "decoy")


@dataclass(frozen=True)
class CalibrationItem:
    question: QuestionRecord
    kind: str
    answer: str
    expected: int

    @property
    def key(self) -> str:
        return f"{self.question.id}|{self.kind}"

    @property
    def prompt(self) -> str:
        return ANSWER_PROMPT.format(
            question=self.question.question,
            expected=self.question.answer,
            generated=self.answer,
        )


def calibration_items(questions: Sequence[QuestionRecord]) -> list[CalibrationItem]:
    scorable = [q for q in questions if normalise_answer(q.answer)]
    items = []
    for index, question in enumerate(scorable):
        own = normalise_answer(question.answer)
        decoy = None
        for step in range(1, len(scorable)):
            candidate = scorable[(index + step) % len(scorable)]
            if normalise_answer(candidate.answer) != own:
                decoy = candidate
                break
        if decoy is None:
            raise ConfigurationError("every question has the same reference answer")
        items.append(CalibrationItem(question, "reference", question.answer, 1))
        items.append(CalibrationItem(question, "decoy", decoy.answer, 0))
    return items


def run_calibration(
    items: Sequence[CalibrationItem],
    client: ModelClient,
    judge: str,
    output: str | Path,
) -> dict[str, Any]:
    """Judge every item not already in ``output`` and summarise all of them."""
    path = Path(output)
    done: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    entry = json.loads(line)
                    done[entry["key"]] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for item in items:
            if item.key in done:
                continue
            value, call = client.judge(judge, item.prompt)
            entry = {
                "key": item.key,
                "question_id": item.question.id,
                "kind": item.kind,
                "expected": item.expected,
                "value": value,
                "judge": judge,
                "cost_usd": call.cost_usd,
                "reply": call.text,
            }
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            done[item.key] = entry
    return summarise(done[item.key] for item in items)


def summarise(entries: Any) -> dict[str, Any]:
    rows = list(entries)
    by_kind = {kind: [r for r in rows if r["kind"] == kind] for kind in KINDS}
    references, decoys = by_kind["reference"], by_kind["decoy"]
    return {
        "items": len(rows),
        "false_negative_rate": (
            sum(r["value"] == 0 for r in references) / len(references)
            if references
            else None
        ),
        "false_positive_rate": (
            sum(r["value"] == 1 for r in decoys) / len(decoys) if decoys else None
        ),
        "cost_usd": sum(float(r["cost_usd"]) for r in rows),
    }


__all__ = ["CalibrationItem", "calibration_items", "run_calibration", "summarise"]
