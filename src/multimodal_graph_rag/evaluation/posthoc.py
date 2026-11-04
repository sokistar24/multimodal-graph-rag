"""Judging passes and evidence states computed over saved run records.

Every pass reads a trace file (one ``RunRecord`` per question and system) and
never regenerates an answer, so it can run after the evaluation, be resumed
after an interruption, and be priced with a dry run first. Verdicts are
appended to a judgements file beside the trace, one JSON line each, keyed by
run, question, system, pass, and repetition.

Passes (plan sections 7.2 and 7.4):

    second-accuracy       a second judge grades accuracy on a stratified sample
                          (or every record) for judge-versus-judge agreement
    second-faithfulness   a second vision judge grades faithfulness on a
                          stratified sample of records that sent an image
    adjudicate            a third judge grades accuracy wherever the primary
                          and second accuracy verdicts disagree; its verdict
                          is final for that record
    sufficiency           for every correct answer that did not receive all the
                          gold evidence, a judge decides whether the reference
                          answer is derivable from the supplied context alone
                          (model-judged, never presented as measured)
    stability             the primary judge re-grades a stratified sample
                          several times; the flip rate measures its stability

``assign_states`` then combines a trace, its judgements, and the closed-book
arm of the same run into the evidence states of
:mod:`..evaluation.attribution`.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..clients import JUDGE_MAX_TOKENS, ModelClient
from ..errors import ConfigurationError
from ..schemas import IMAGE_SUFFIXES, QuestionRecord, canonical_json
from .answer_match import exact_match
from .attribution import EvidenceState, classify_evidence_state
from .judges import ANSWER_PROMPT, VERDICT_INSTRUCTION, VISION_FAITHFULNESS_PROMPT

PASSES = (
    "second-accuracy",
    "second-faithfulness",
    "adjudicate",
    "sufficiency",
    "stability",
)
CLOSED_BOOK = "control:closed-book"
# Arms with no evidence to be faithful to or sufficient from.
NO_CONTEXT_ARMS = frozenset({CLOSED_BOOK})

SUFFICIENCY_PROMPT = (
    "Decide whether the EXPECTED answer to the QUESTION can be derived from the "
    "CONTEXT (and the attached image, if any) alone, without outside "
    "knowledge.\n\n"
    "QUESTION: {question}\nEXPECTED: {expected}\n\nCONTEXT:\n{context}\n\n"
    + VERDICT_INSTRUCTION.format(one="1 (derivable)", zero="0 (not derivable)")
)


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    trace = Path(path)
    if not trace.is_file():
        raise ConfigurationError(f"trace does not exist: {trace}")
    with trace.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def judgements_path(trace: str | Path) -> Path:
    path = Path(trace)
    return path.with_name(path.name.replace("trace_", "judgements_", 1))


def judge_context(record: Mapping[str, Any]) -> str:
    """The evidence as the faithfulness judge saw it during the run.

    Mirrors ``judge_context`` in :mod:`..pipelines.systems`: text passages in
    rank order, then graph facts, then the captions of the images sent.
    """
    items = record["evidence"]["items"]
    parts = [i["content"] for i in items if i["modality"] == "text"]
    parts += [i["content"] for i in items if i["modality"] == "graph"]
    parts += [i["content"] for i in items if _sent(i)]
    return "\n".join(part for part in parts if part)


def sent_images(record: Mapping[str, Any]) -> list[str]:
    return [i["source"] for i in record["evidence"]["items"] if _sent(i)]


def _sent(item: Mapping[str, Any]) -> bool:
    return item["modality"] == "image" and str(item["selection_reason"]).endswith(
        "(sent)"
    )


def gold_received(record: Mapping[str, Any], question: QuestionRecord) -> int:
    """Gold units that reached the generator: text sources, or images sent."""
    items = record["evidence"]["items"]
    text = {i["source"] for i in items if i["modality"] == "text"}
    images = set(sent_images(record))
    return sum(
        1
        for gold in question.gold_sources
        if (gold in images if gold.lower().endswith(IMAGE_SUFFIXES) else gold in text)
    )


def stratified_sample(
    records: Sequence[Mapping[str, Any]], fraction: float, seed: int
) -> list[Mapping[str, Any]]:
    """A deterministic sample of ``fraction`` of the records of every system."""
    if not 0 < fraction <= 1:
        raise ConfigurationError("fraction must be in (0, 1]")
    by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_system[record["system"]].append(record)
    chosen: list[Mapping[str, Any]] = []
    for system in sorted(by_system):
        ranked = sorted(
            by_system[system],
            key=lambda r: hashlib.sha256(
                f"{seed}:{r['question_id']}:{system}".encode()
            ).hexdigest(),
        )
        chosen.extend(ranked[: max(1, round(fraction * len(ranked)))])
    return chosen


@dataclass(frozen=True)
class Task:
    record: Mapping[str, Any]
    pass_name: str
    repeat: int = 0

    @property
    def key(self) -> str:
        r = self.record
        return f"{r['run_id']}|{r['question_id']}|{r['system']}|{self.pass_name}|{self.repeat}"


class JudgementStore:
    """Append-only verdicts for one trace; a key already present is skipped."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        entry = json.loads(line)
                        self.entries[entry["key"]] = entry

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def add(self, entry: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(dict(entry)) + "\n")
        self.entries[entry["key"]] = dict(entry)

    def values(self, pass_name: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
        found: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for entry in self.entries.values():
            if entry["pass"] == pass_name:
                found[(entry["question_id"], entry["system"])].append(entry)
        return found


def primary_accuracy(record: Mapping[str, Any]) -> int:
    return int(record["raw_judges"]["accuracy"]["value"])


def is_correct(
    record: Mapping[str, Any],
    question: QuestionRecord,
    store: JudgementStore,
    correctness: str,
) -> bool:
    """Exact match, or the judge verdict with any adjudication applied."""
    if correctness == "em":
        return bool(exact_match(record["answer"], question.answer))
    if correctness != "judge":
        raise ConfigurationError("correctness must be 'judge' or 'em'")
    adjudicated = store.values("adjudicate").get(
        (record["question_id"], record["system"])
    )
    if adjudicated:
        return bool(adjudicated[0]["value"])
    return bool(primary_accuracy(record))


def plan_tasks(
    pass_name: str,
    records: Sequence[Mapping[str, Any]],
    questions: Mapping[str, QuestionRecord],
    store: JudgementStore,
    *,
    fraction: float = 1.0,
    seed: int = 20260903,
    repeats: int = 3,
    correctness: str = "judge",
) -> list[Task]:
    """The calls a pass still has to make (records already judged are skipped)."""
    if pass_name not in PASSES:
        raise ConfigurationError(f"unknown pass {pass_name!r}; choose from {PASSES}")
    pool: Iterable[Mapping[str, Any]]
    if pass_name == "second-accuracy":
        pool = stratified_sample(records, fraction, seed)
        tasks = [Task(r, pass_name) for r in pool]
    elif pass_name == "second-faithfulness":
        with_image = [
            r for r in records if sent_images(r) and r["system"] not in NO_CONTEXT_ARMS
        ]
        tasks = (
            [Task(r, pass_name) for r in stratified_sample(with_image, fraction, seed)]
            if with_image
            else []
        )
    elif pass_name == "adjudicate":
        second = store.values("second-accuracy")
        tasks = [
            Task(r, pass_name)
            for r in records
            if (r["question_id"], r["system"]) in second
            and second[(r["question_id"], r["system"])][0]["value"]
            != primary_accuracy(r)
        ]
    elif pass_name == "sufficiency":
        tasks = [
            Task(r, pass_name)
            for r in records
            if r["system"] not in NO_CONTEXT_ARMS
            and is_correct(r, questions[r["question_id"]], store, correctness)
            and gold_received(r, questions[r["question_id"]])
            < len(questions[r["question_id"]].gold_sources)
        ]
    else:  # stability
        pool = stratified_sample(records, fraction, seed)
        tasks = [
            Task(r, pass_name, repeat) for r in pool for repeat in range(1, repeats + 1)
        ]
    return [task for task in tasks if task.key not in store]


def run_task(
    task: Task,
    question: QuestionRecord,
    client: ModelClient,
    judge: str,
    image_dir: str | Path | None,
    vision_mode: str,
) -> dict[str, Any]:
    record = task.record
    images: list[Path] = []
    if vision_mode == "pixels" and image_dir is not None:
        images = [Path(image_dir) / name for name in sent_images(record)]
    if task.pass_name in ("second-accuracy", "adjudicate", "stability"):
        prompt = ANSWER_PROMPT.format(
            question=question.question,
            expected=question.answer,
            generated=record["answer"],
        )
        images = []
    elif task.pass_name == "second-faithfulness":
        prompt = VISION_FAITHFULNESS_PROMPT.format(
            context=judge_context(record), generated=record["answer"]
        )
    else:
        prompt = SUFFICIENCY_PROMPT.format(
            question=question.question,
            expected=question.answer,
            context=judge_context(record) or "(none)",
        )
    value, call = client.judge(judge, prompt, images=images)
    lines = [line for line in call.text.splitlines() if line.strip()]
    return {
        "key": task.key,
        "run_id": record["run_id"],
        "question_id": record["question_id"],
        "system": record["system"],
        "pass": task.pass_name,
        "repeat": task.repeat,
        "judge": judge,
        "value": value,
        "justification": "\n".join(lines[:-1]).strip(),
        "response": call.text,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "cost_usd": call.cost_usd,
        "max_tokens": JUDGE_MAX_TOKENS,
    }


def assign_states(
    records: Sequence[Mapping[str, Any]],
    questions: Mapping[str, QuestionRecord],
    store: JudgementStore,
    *,
    available_sources: set[str],
    correctness: str = "judge",
) -> list[dict[str, Any]]:
    """One evidence state per record, following Table 2 of the paper.

    A correct answer that received incomplete gold evidence and has no
    sufficiency verdict yet is reported as ``pending_sufficiency`` rather than
    defaulted, so no state is assigned without the judgement it needs.
    """
    closed_book = {
        r["question_id"]: is_correct(r, questions[r["question_id"]], store, correctness)
        for r in records
        if r["system"] == CLOSED_BOOK
    }
    sufficiency = store.values("sufficiency")
    rows = []
    for record in records:
        if record["system"] == CLOSED_BOOK:
            continue
        question = questions[record["question_id"]]
        correct = is_correct(record, question, store, correctness)
        found = gold_received(record, question)
        required = len(question.gold_sources)
        verdict = sufficiency.get((record["question_id"], record["system"]))
        if correct and found < required and verdict is None:
            state = "pending_sufficiency"
        else:
            state = classify_evidence_state(
                evidence_available=all(
                    gold in available_sources for gold in question.gold_sources
                ),
                gold_found=found,
                gold_required=required,
                answer_correct=correct,
                non_gold_support=bool(verdict and verdict[0]["value"]),
                closed_book_correct=closed_book.get(record["question_id"], False),
            ).value
        rows.append(
            {
                "question_id": record["question_id"],
                "system": record["system"],
                "correct": int(correct),
                "gold_received": found,
                "gold_required": required,
                "state": state,
            }
        )
    return rows


__all__ = [
    "EvidenceState",
    "JudgementStore",
    "PASSES",
    "Task",
    "assign_states",
    "gold_received",
    "judge_context",
    "judgements_path",
    "load_trace",
    "plan_tasks",
    "run_task",
    "sent_images",
    "stratified_sample",
]
