"""Audit instruments: graph-triple sampling and visual-question leakage screening.

Both produce sheets for human reviewers. The leakage screen labels a clean
programmatic result ``screened_unflagged``, never ``verified``; semantic
entailment checks and human review are separate protocol stages.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from ..errors import ConfigurationError
from ..schemas import load_questions
from .integrity import audit_answer_recoverability

TRIPLE_AUDIT_FIELDS = (
    "audit_id",
    "corpus",
    "cache_key",
    "subject",
    "relation",
    "object",
    "extraction_frequency",
    "triple_correct",
    "source_entails",
    "subject_canonical",
    "object_canonical",
    "relation_specific",
    "provenance_correct",
    "reviewer",
    "notes",
)
FIGURE_AUDIT_FIELDS = (
    "question_id",
    "source",
    "construction_method",
    "caption_exact",
    "caption_numeric",
    "corpus_exact",
    "corpus_numeric",
    "audit_status",
)


def sample_triple_audit(
    cache_path: str | Path, corpus: str, count: int, seed: int
) -> list[dict[str, object]]:
    """Deterministic sample of unique triples, half frequency-stratified."""
    if count < 4:
        raise ConfigurationError("count must be at least 4")
    raw = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    occurrences: dict[tuple[str, str, str], list[str]] = {}
    for key, triples in raw.items():
        for triple in triples:
            if len(triple) >= 3:
                value = tuple(str(part) for part in triple[:3])
                occurrences.setdefault(value, []).append(key)
    ranked = sorted(occurrences, key=lambda triple: (-len(occurrences[triple]), triple))
    if count > len(ranked):
        raise ConfigurationError(
            f"requested {count} triples but the cache holds {len(ranked)} unique triples"
        )
    bins = [ranked[i::4] for i in range(4)]
    rng = random.Random(seed)
    chosen: list[tuple[str, str, str]] = []
    per_bin = count // 8
    for group in bins:
        chosen.extend(rng.sample(group, min(per_bin, len(group))))
    already = set(chosen)
    remaining = [triple for triple in ranked if triple not in already]
    chosen.extend(rng.sample(remaining, count - len(chosen)))
    rows = []
    for index, (subject, relation, obj) in enumerate(chosen, 1):
        rows.append(
            {
                "audit_id": f"{corpus}-{index:04d}",
                "corpus": corpus,
                "cache_key": occurrences[(subject, relation, obj)][0],
                "subject": subject,
                "relation": relation,
                "object": obj,
                "extraction_frequency": len(occurrences[(subject, relation, obj)]),
                **{field: "" for field in TRIPLE_AUDIT_FIELDS[7:]},
            }
        )
    return rows


def _corpus_text(corpus_dir: Path, image_name: str) -> str:
    paper = image_name.split("-Figure", 1)[0].split("-Table", 1)[0]
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(corpus_dir.glob(f"{paper}*.txt"))
    )


def figure_integrity_rows(
    question_file: str | Path, captions_file: str | Path, corpus_dir: str | Path
) -> list[dict[str, object]]:
    """Screen every visual question for answer recoverability in text channels."""
    captions = json.loads(Path(captions_file).read_text(encoding="utf-8"))
    corpus = Path(corpus_dir)
    rows: list[dict[str, object]] = []
    for question in load_questions(question_file):
        source = question.gold_sources[0]
        caption_audit = audit_answer_recoverability(
            question.answer, str(captions.get(source, ""))
        )
        corpus_audit = audit_answer_recoverability(
            question.answer, _corpus_text(corpus, source)
        )
        flagged = caption_audit.needs_human_review or corpus_audit.needs_human_review
        rows.append(
            {
                "question_id": question.id,
                "source": source,
                "construction_method": question.construction_method,
                "caption_exact": caption_audit.exact_match,
                "caption_numeric": caption_audit.numeric_equivalent,
                "corpus_exact": corpus_audit.exact_match,
                "corpus_numeric": corpus_audit.numeric_equivalent,
                "audit_status": "flagged_for_human_review"
                if flagged
                else "screened_unflagged",
            }
        )
    return rows


def paired_accuracy(
    detail_rows: Sequence[dict[str, str]], baseline: str, treatment: str
) -> tuple[list[int], list[int]]:
    """Per-question accuracy vectors for a paired system contrast."""
    left_col, right_col = f"{baseline}_acc", f"{treatment}_acc"
    if (
        not detail_rows
        or left_col not in detail_rows[0]
        or right_col not in detail_rows[0]
    ):
        raise ConfigurationError(f"detail rows lack {left_col} or {right_col}")
    left = [int(float(row[left_col])) for row in detail_rows]
    right = [int(float(row[right_col])) for row in detail_rows]
    return left, right
