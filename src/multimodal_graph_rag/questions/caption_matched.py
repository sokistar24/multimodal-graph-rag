"""Caption-answerable figure questions matched crop-for-crop to the pixel set.

Same author (Claude Haiku), same crops, different information: the author sees
only the pipeline caption, never the crop. Holding author and crops fixed
isolates the effect of question construction on apparent multimodal gains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..clients import ModelClient
from ..errors import ConfigurationError
from ..evaluation.integrity import audit_answer_recoverability
from ..schemas import QuestionRecord, content_hash, load_questions, write_questions
from .authoring import FIGURE_AUTHOR
from .validators import parse_json_array, rejection_reason

CONSTRUCTION = "claude-haiku-caption-authored"

PROMPT = """Below is a one-sentence description of a visual region from a scientific paper.

Write ONE question answerable using ONLY the information in this description,
plus its short answer.

RULE 1 — SELF-CONTAINED AND SPECIFIC:
The question will be asked against a corpus of scientific papers. It must name
the specific method, quantity, variable, or comparison involved, so that it is
unambiguous. Never write "the study", "the results", "the data" as bare
references.

RULE 2 — DO NOT REVEAL WHERE THE ANSWER LIVES:
The question must NOT contain: figure, table, chart, image, graph, diagram,
plot, panel, shown, depicted, described, illustrated.

RULE 3 — GROUNDED:
The answer must be expressible from the description alone. Invent nothing.

If the description is too vague to support a specific question, reply with an
empty array: []

DESCRIPTION:
\"\"\"
{caption}
\"\"\"

Respond with ONLY a JSON array containing exactly one object, no markdown fences:
[{{"q": "...", "answer": "..."}}]"""


@dataclass(frozen=True)
class CaptionMatchedReport:
    written: int
    crops: int
    skipped: dict[str, int]
    caption_answers_in_caption: int
    pixel_answers_in_caption: int
    output: Path


def author_caption_matched(
    client: ModelClient,
    *,
    pixel_file: str | Path,
    captions_file: str | Path,
    out_path: str | Path,
) -> CaptionMatchedReport:
    pixel_questions = load_questions(pixel_file)
    with Path(captions_file).open(encoding="utf-8") as stream:
        captions = json.load(stream)
    output = Path(out_path)
    records = load_questions(output) if output.is_file() else []
    done = {record.gold_sources[0] for record in records}
    skipped: dict[str, int] = {}

    for pixel in pixel_questions:
        crop = pixel.gold_sources[0]
        if crop in done:
            continue
        caption = captions.get(crop)
        if not caption:
            skipped["no pipeline caption"] = skipped.get("no pipeline caption", 0) + 1
            continue
        result = client.call(
            FIGURE_AUTHOR, user=PROMPT.format(caption=caption), max_tokens=400
        )
        parsed = parse_json_array(result.text)
        if not parsed or not isinstance(parsed[0], dict):
            skipped["author declined or unparseable"] = (
                skipped.get("author declined or unparseable", 0) + 1
            )
            continue
        question = str(parsed[0].get("q", "")).strip()
        answer = str(parsed[0].get("answer", "")).strip()
        reason = rejection_reason(question) if question and answer else "malformed item"
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        records.append(
            QuestionRecord(
                id=content_hash({"q": question, "answer": answer, "sources": [crop]})[
                    :16
                ],
                question=question,
                answer=answer,
                gold_sources=(crop,),
                question_type="figure",
                construction_method=CONSTRUCTION,
                evidence_requirement="caption",
                metadata={"matched_pixel_question_id": pixel.id},
            )
        )
        write_questions(records, output, source_file=Path(pixel_file).name)
        print(f"[ok] {crop}: {question}")

    if not records:
        raise ConfigurationError("no caption-matched questions were produced")

    def answers_in_caption(items: list[QuestionRecord]) -> int:
        return sum(
            audit_answer_recoverability(
                item.answer, str(captions.get(item.gold_sources[0], ""))
            ).exact_match
            for item in items
        )

    return CaptionMatchedReport(
        written=len(records),
        crops=len(pixel_questions),
        skipped=skipped,
        caption_answers_in_caption=answers_in_caption(records),
        pixel_answers_in_caption=answers_in_caption(pixel_questions),
        output=output,
    )
