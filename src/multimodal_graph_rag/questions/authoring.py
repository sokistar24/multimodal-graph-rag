"""Author single-corpus question sets: text, within-page multi-hop, and figure.

Both authors sit outside every generator family under test: DeepSeek writes
text and multi-hop questions from page text, and Claude Haiku writes figure
questions from the crop itself and then re-verifies each one against the same
crop with a fresh call. Questions that fail a validator are counted as
rejections and another page or crop is tried; provider failures are raised.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..clients import ModelClient
from ..errors import ConfigurationError
from ..retrieval.visual import image_files
from ..schemas import QuestionRecord, content_hash, load_questions, write_questions
from .validators import parse_json_array, rejection_reason

TEXT_AUTHOR = "deepseek"
FIGURE_AUTHOR = "claude-haiku"
DEFAULT_SEED = 42
DEFAULT_TARGETS = {"text": 35, "multihop": 30, "figure": 35}
MIN_PAGE_CHARS = 400
MIN_MULTIHOP_CHARS = 900
ATTEMPT_FACTOR = 6
CONSTRUCTION = {
    "text": "deepseek-single-passage",
    "multihop": "deepseek-two-fact-same-page",
    "figure": "claude-haiku-crop-authored",
}

TEXT_FEWSHOT = """Examples of GOOD questions — each names a specific entity, so exactly one document can answer it:

{"q": "What method is used to assess the spatial consistency of ERP maps across subjects?", "answer": "the grand mean ERP map"}
{"q": "What is of paramount importance for the activation of p70S6K?", "answer": "the phosphorylation of Serine residue in 411 position"}
{"q": "What vaccine is currently the only one used in China for leptospirosis?", "answer": "multivalent inactivated vaccine"}

Examples of BAD questions — grammatically fine, but hundreds of papers could answer them:

{"q": "How many people attended the workshop?"}          <- WHICH workshop?
{"q": "What is the aim of the current paper?"}            <- WHICH paper?
{"q": "How many participants were in the study?"}         <- WHICH study?"""

MULTIHOP_FEWSHOT = """Examples of GOOD multi-hop questions — each needs TWO separate facts AND names specific entities:

{"q": "What is the odds ratio for headache in individuals with chronic pain, and what is it for back pain?", "answer": "1.83 (1.36-2.46) for chronic pain, 2.72 (1.73-4.29) for back pain"}
{"q": "Which treatment group showed the greatest reduction in HbA1c, and what dose did that group receive?", "answer": "the high-dose metformin group, receiving 2000 mg daily"}

Examples of BAD questions — hundreds of papers could answer these:

{"q": "How many participants were in the study, and what was their mean age?"}   <- WHICH study?
{"q": "What did the authors conclude, and how many patients were enrolled?"}     <- WHICH authors?"""

TEXT_INSTRUCTION = (
    "Write ONE question answerable from a SINGLE sentence or passage in the text below."
)
MULTIHOP_INSTRUCTION = (
    "Write ONE question that requires combining TWO SEPARATE facts from different "
    "parts of the passage. A reader must find both to answer. Do not write a "
    "question answerable from a single sentence."
)

TEXT_PROMPT = """{instruction}

RULE 1 — SELF-CONTAINED AND SPECIFIC (most important):
The question will be asked against a corpus of 1,000 unrelated scientific pages.
It must name the specific condition, cohort, molecule, method, place, or entity
involved, so that exactly ONE page can answer it.
- NEVER write "the study", "the paper", "this trial", "the workshop",
  "the participants", "the authors", "the current research" or similar bare
  references. The reader has no idea which document you mean.
- Instead, name the actual subject: not "the participants in the study" but
  "post-stroke patients", not "the workshop" but "the WHO malaria surveillance
  workshop".
- Do NOT copy a whole sentence from the passage. Rephrase in your own words
  while keeping the specific entity names.
- NEVER write "this work", "the passage", "the present paper", or "the
  proposed loss/method/model" — name the actual method, model, or subject.
- NEVER include a page identifier or arXiv id in the question.
- Keep the question under 30 words. Two short facts, not two welded questions.

RULE 2 — DO NOT REVEAL WHERE THE ANSWER LIVES:
The question must NOT contain: figure, table, chart, image, graph, diagram,
shown, depicted, above, below.

RULE 3 — GROUNDED:
The answer must be short and directly supported by the passage. Invent nothing.

If the passage is too garbled or generic to support a specific question, reply
with an empty array: []

{fewshot}

PASSAGE (from page {page_id}):
\"\"\"
{page_text}
\"\"\"

Respond with ONLY a JSON array containing exactly one object, no markdown fences:
[{{"q": "...", "answer": "..."}}]"""

FIGURE_PROMPT = """Look at this figure or table from a scientific paper.

Write ONE question that can ONLY be answered by looking at it — a fact that
lives in the visual content, not in surrounding prose.

RULE 1 — SELF-CONTAINED AND SPECIFIC (most important):
The question will be asked against a corpus of 1,000 unrelated scientific pages
and their figures. It must name the specific variable, group, condition, or
measurement involved, so exactly ONE image can answer it.
- NEVER write "the study", "the table", "this trial", "the participants",
  "the interviews" or similar bare references — the reader has no idea which
  document you mean, and hundreds of pages would match.
- Name the actual subject visible in the image: the variable on the axis, the
  named treatment arm, the specific cohort, the labelled group.
  BAD:  "How many female students were included in the in-depth interviews?"
  GOOD: "How many female nursing students were surveyed about needle-stick
         injury reporting?"
  BAD:  "What does it show about glucose levels?"
  GOOD: "What was the mean plasma glucose level at 120 minutes in the
         metformin arm?"

RULE 2 — DO NOT REVEAL WHERE THE ANSWER LIVES:
The question must NOT contain: figure, table, chart, image, graph, diagram,
panel, shown, depicted, above, below.

RULE 3 — GROUNDED:
The answer must be short, concrete, and readable directly from the visual.

If the content is too vague or unlabelled to support a specific, uniquely
identifiable question, reply with an empty array: []

Respond with ONLY a JSON array containing exactly one object, no markdown fences:
[{"q": "...", "answer": "..."}]"""

VERIFY_PROMPT = (
    "Question: {question}\nProposed answer: {answer}\n\n"
    "Can this question be answered from this image, and is the proposed answer "
    "correct according to it?\nReply with ONLY the digit 1 (yes) or 0 (no):"
)


@dataclass(frozen=True)
class AuthoringReport:
    kind: str
    written: int
    target: int
    attempts: int
    rejected: dict[str, int]
    output: Path


def _draft(
    client: ModelClient, author: str, prompt: str, images: tuple[Path, ...] = ()
) -> tuple[str, str] | str:
    """A ``(question, answer)`` pair, or a rejection reason string."""
    result = client.call(author, user=prompt, images=images, max_tokens=400)
    parsed = parse_json_array(result.text)
    if parsed is None:
        return "unparseable reply"
    if not parsed:
        return "author declined"
    item = parsed[0]
    if (
        not isinstance(item, dict)
        or not str(item.get("q", "")).strip()
        or not str(item.get("answer", "")).strip()
    ):
        return "malformed item"
    question, answer = str(item["q"]).strip(), str(item["answer"]).strip()
    reason = rejection_reason(question)
    return reason if reason else (question, answer)


def _record(question: str, answer: str, source: str, kind: str) -> QuestionRecord:
    return QuestionRecord(
        id=content_hash({"q": question, "answer": answer, "sources": [source]})[:16],
        question=question,
        answer=answer,
        gold_sources=(source,),
        question_type=kind,
        construction_method=CONSTRUCTION[kind],
        evidence_requirement="visual" if kind == "figure" else "text",
    )


def _existing(out_path: Path) -> list[QuestionRecord]:
    return load_questions(out_path) if out_path.is_file() else []


def load_pages(corpus_dir: Path, min_chars: int, seed: int) -> list[tuple[str, str]]:
    pages = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) >= min_chars:
            pages.append((path.stem, text))
    if not pages:
        raise ConfigurationError(
            f"no pages with at least {min_chars} characters in {corpus_dir}"
        )
    random.Random(seed).shuffle(pages)
    return pages


def _author_loop(
    kind: str,
    target: int,
    candidates: list,
    out_path: Path,
    used: set[str],
    attempt: Callable[[object], tuple[QuestionRecord, str] | str],
    *,
    smoke: bool,
) -> AuthoringReport:
    items = _existing(out_path)
    rejected: dict[str, int] = {}
    attempts = 0
    for candidate in candidates:
        if len(items) >= target:
            break
        outcome = attempt(candidate)
        if isinstance(outcome, str):
            if outcome == "already used":
                continue
            attempts += 1
            rejected[outcome] = rejected.get(outcome, 0) + 1
        else:
            attempts += 1
            record, source = outcome
            items.append(record)
            used.add(source)
            write_questions(items, out_path, source_file=out_path.name)
            if smoke or len(items) % 5 == 0:
                print(f"  [{len(items)}/{target}] {source}: {record.question[:80]}")
        if attempts > target * ATTEMPT_FACTOR:
            print(
                f"  stopping: {sum(rejected.values())} rejections in {attempts} attempts"
            )
            break
    return AuthoringReport(kind, len(items), target, attempts, rejected, out_path)


def author_text_questions(
    client: ModelClient,
    *,
    corpus_dir: str | Path,
    kind: str,
    target: int,
    out_path: str | Path,
    seed: int = DEFAULT_SEED,
    smoke: bool = False,
) -> AuthoringReport:
    if kind not in ("text", "multihop"):
        raise ConfigurationError("kind must be 'text' or 'multihop'")
    output = Path(out_path)
    used = {q.gold_sources[0] for q in _existing(output)}
    min_chars = MIN_MULTIHOP_CHARS if kind == "multihop" else MIN_PAGE_CHARS
    pages = load_pages(Path(corpus_dir), min_chars, seed)
    instruction = MULTIHOP_INSTRUCTION if kind == "multihop" else TEXT_INSTRUCTION
    fewshot = MULTIHOP_FEWSHOT if kind == "multihop" else TEXT_FEWSHOT

    def attempt(page: tuple[str, str]) -> tuple[QuestionRecord, str] | str:
        page_id, text = page
        source = f"{page_id}.txt"
        if source in used:
            return "already used"
        prompt = TEXT_PROMPT.format(
            instruction=instruction,
            fewshot=fewshot,
            page_id=page_id,
            page_text=text[:4000],
        )
        draft = _draft(client, TEXT_AUTHOR, prompt)
        if isinstance(draft, str):
            return draft
        return _record(draft[0], draft[1], source, kind), source

    return _author_loop(kind, target, pages, output, used, attempt, smoke=smoke)


def verify_figure_question(
    client: ModelClient, crop: Path, question: str, answer: str
) -> bool:
    verdict, _ = client.judge(
        FIGURE_AUTHOR,
        VERIFY_PROMPT.format(question=question, answer=answer),
        images=[crop],
    )
    return verdict == 1


def author_figure_questions(
    client: ModelClient,
    *,
    image_dir: str | Path,
    target: int,
    out_path: str | Path,
    seed: int = DEFAULT_SEED,
    smoke: bool = False,
) -> AuthoringReport:
    output = Path(out_path)
    used = {q.gold_sources[0] for q in _existing(output)}
    crops = image_files(image_dir)
    random.Random(seed).shuffle(crops)

    def attempt(crop: Path) -> tuple[QuestionRecord, str] | str:
        if crop.name in used:
            return "already used"
        draft = _draft(client, FIGURE_AUTHOR, FIGURE_PROMPT, images=(crop,))
        if isinstance(draft, str):
            return draft
        if not verify_figure_question(client, crop, draft[0], draft[1]):
            return "failed self-verification"
        return _record(draft[0], draft[1], crop.name, "figure"), crop.name

    return _author_loop("figure", target, crops, output, used, attempt, smoke=smoke)
