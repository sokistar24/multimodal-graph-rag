"""Convert SPIQA's native human-curated QA into an evaluable question set.

`rag ingest --config <spiqa>` already saves the QA that SPIQA's curators wrote
for every sampled paper. Those items are human-authored, human-answered, and
carry human figure-level provenance: the `reference` field names the exact crop
the answer was read from. That makes them the one visual set in this project
whose questions were not written by a model, which removes the circularity of
the LLM-authored figure sets (the same model authored a question from a crop
and then verified its own question against that crop).

This converter changes no content. It attaches a stable id, records the
construction method as human-authored, and runs the leakage screen against both
caption channels so that construction intent and screened leakage status stay
separate variables. Nothing is dropped for being flagged; flagged items are
labelled so the tables can tier them.

The project's authoring validators are deliberately NOT applied. They exist to
constrain a model that is writing questions — rejecting an orphan reference or a
modality cue. Applying them here would silently edit a published benchmark. The
report counts how many native questions name their modality instead, because
that is a genuine methodological difference between this set and the authored
ones and belongs in the write-up rather than in a filter.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigurationError
from ..evaluation.integrity import audit_answer_recoverability
from ..schemas import QuestionRecord, content_hash, write_questions
from .validators import leaks_modality

CONSTRUCTION_METHOD = "spiqa_native_human"
QUESTION_TYPE = "figure_native"
REQUIRED_FIELDS = ("question", "answer", "reference", "paper_id")


@dataclass(frozen=True)
class ConversionReport:
    kind: str
    written: int
    available: int
    skipped_missing_crop: int
    skipped_incomplete: int
    flagged_pipeline_caption: int
    flagged_author_caption: int
    names_its_modality: int
    output: Path


def _load_json(path: str | Path, label: str) -> object:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigurationError(f"{label} does not exist: {file_path}")
    with file_path.open(encoding="utf-8") as stream:
        return json.load(stream)


def convert_spiqa_native(
    gold_file: str | Path,
    captions_file: str | Path,
    out_path: str | Path,
    *,
    author_captions_file: str | Path | None = None,
    target: int | None = None,
    seed: int = 42,
    exclude_flagged: bool = False,
) -> ConversionReport:
    """Write a question set from SPIQA's curated QA for the ingested crops.

    Items whose referenced crop was not ingested are skipped, so the set always
    matches the corpus it will be evaluated against. ``target`` takes a
    deterministic uniform sample; ``exclude_flagged`` restricts that sample to
    items the leakage screen did not flag in either caption channel.
    """
    gold = _load_json(gold_file, "SPIQA gold QA file")
    if not isinstance(gold, list) or not gold:
        raise ConfigurationError(
            f"SPIQA gold QA file is empty or not a list: {gold_file}"
        )
    captions = _load_json(captions_file, "captions file")
    if not isinstance(captions, dict):
        raise ConfigurationError(f"captions file is not an object: {captions_file}")
    author_captions: dict[str, str] = {}
    if author_captions_file and Path(author_captions_file).is_file():
        loaded = _load_json(author_captions_file, "author captions file")
        if isinstance(loaded, dict):
            author_captions = {str(k): str(v) for k, v in loaded.items()}

    records: list[QuestionRecord] = []
    skipped_missing_crop = skipped_incomplete = 0
    flagged_pipeline = flagged_author = names_modality = 0

    for item in gold:
        if not isinstance(item, dict) or any(
            not str(item.get(field, "")).strip() for field in REQUIRED_FIELDS
        ):
            skipped_incomplete += 1
            continue
        reference = str(item["reference"])
        if reference not in captions:
            skipped_missing_crop += 1
            continue
        question = str(item["question"]).strip()
        answer = str(item["answer"]).strip()

        pipeline = audit_answer_recoverability(answer, str(captions[reference]))
        author = audit_answer_recoverability(answer, author_captions.get(reference, ""))
        flagged = pipeline.needs_human_review or author.needs_human_review
        flagged_pipeline += pipeline.needs_human_review
        flagged_author += author.needs_human_review
        mentions_modality = leaks_modality(question)
        names_modality += mentions_modality

        records.append(
            QuestionRecord(
                id=content_hash({"spiqa": reference, "q": question})[:16],
                question=question,
                answer=answer,
                gold_sources=(reference,),
                question_type=QUESTION_TYPE,
                construction_method=CONSTRUCTION_METHOD,
                evidence_requirement="visual",
                graph_seeded=False,
                leakage_status=(
                    "flagged_for_human_review" if flagged else "screened_unflagged"
                ),
                metadata={
                    "paper_id": str(item["paper_id"]),
                    "spiqa_explanation": str(item.get("explanation", "")),
                    "caption_exact": pipeline.exact_match,
                    "caption_numeric": pipeline.numeric_equivalent,
                    "author_caption_exact": author.exact_match,
                    "author_caption_numeric": author.numeric_equivalent,
                    "names_its_modality": mentions_modality,
                },
            )
        )

    if not records:
        raise ConfigurationError(
            f"no SPIQA gold QA item referenced an ingested crop in {captions_file}"
        )

    available = len(records)
    pool = (
        [r for r in records if r.leakage_status == "screened_unflagged"]
        if exclude_flagged
        else records
    )
    if exclude_flagged and not pool:
        raise ConfigurationError("every item was flagged; nothing left to sample")
    if target is not None:
        if target < 1:
            raise ConfigurationError("target must be positive")
        if target > len(pool):
            raise ConfigurationError(
                f"requested {target} questions but only {len(pool)} are available"
            )
        pool = sorted(random.Random(seed).sample(pool, target), key=lambda r: r.id)
    else:
        pool = sorted(pool, key=lambda r: r.id)

    destination = Path(out_path)
    write_questions(pool, destination, source_file=Path(gold_file).name)
    return ConversionReport(
        kind="spiqa-native",
        written=len(pool),
        available=available,
        skipped_missing_crop=skipped_missing_crop,
        skipped_incomplete=skipped_incomplete,
        flagged_pipeline_caption=flagged_pipeline,
        flagged_author_caption=flagged_author,
        names_its_modality=names_modality,
        output=destination,
    )
