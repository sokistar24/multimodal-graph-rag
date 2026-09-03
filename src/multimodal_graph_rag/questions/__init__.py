"""Question authoring protocols and the validators every set must pass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schemas import QuestionRecord, load_questions, write_questions
from .validators import rejection_reason


@dataclass(frozen=True)
class CleaningReport:
    path: Path
    kept: int
    removed: tuple[tuple[str, str], ...]


def clean_question_file(path: str | Path) -> CleaningReport:
    """Drop items that fail the current validators and rewrite the file."""
    file_path = Path(path)
    kept: list[QuestionRecord] = []
    removed: list[tuple[str, str]] = []
    for record in load_questions(file_path):
        reason = rejection_reason(record.question)
        if reason:
            removed.append((reason, record.question))
        else:
            kept.append(record)
    write_questions(kept, file_path, source_file=file_path.name)
    return CleaningReport(file_path, len(kept), tuple(removed))
