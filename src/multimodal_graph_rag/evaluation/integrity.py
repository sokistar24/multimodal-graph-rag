"""Leakage screening that separates construction intent from audit status."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9.]+", text.lower()))


def numeric_values(text: str) -> set[Decimal]:
    values = set()
    for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text):
        try:
            values.add(Decimal(match.replace(",", "")))
        except InvalidOperation:
            continue
    return values


@dataclass(frozen=True)
class LeakageAudit:
    exact_match: bool
    numeric_equivalent: bool
    semantic_score: float | None
    semantic_flag: bool
    needs_human_review: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_answer_recoverability(
    answer: str,
    candidate_text: str,
    *,
    semantic_scorer: Callable[[str, str], float] | None = None,
    semantic_threshold: float = 0.82,
) -> LeakageAudit:
    answer_norm = normalise(answer)
    text_norm = normalise(candidate_text)
    exact = bool(answer_norm and answer_norm in text_norm)
    answer_numbers = numeric_values(answer)
    numeric = bool(answer_numbers and answer_numbers <= numeric_values(candidate_text))
    semantic_score = (
        semantic_scorer(answer, candidate_text) if semantic_scorer else None
    )
    semantic_flag = semantic_score is not None and semantic_score >= semantic_threshold
    flagged = exact or numeric or semantic_flag
    return LeakageAudit(
        exact_match=exact,
        numeric_equivalent=numeric,
        semantic_score=semantic_score,
        semantic_flag=semantic_flag,
        needs_human_review=flagged,
        status="flagged_for_human_review" if flagged else "screened_unflagged",
    )
