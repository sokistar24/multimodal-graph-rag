"""Operational evidence-attribution states used by the revised study."""

from __future__ import annotations

from enum import StrEnum


class EvidenceState(StrEnum):
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    RETRIEVAL_FAILURE = "retrieval_failure"
    INCOMPLETE_RETRIEVAL = "incomplete_retrieval"
    COMPLETE_RETRIEVAL = "complete_retrieval"
    EVIDENCE_USE_FAILURE = "evidence_use_failure"
    NON_GOLD_SUPPORTED = "non_gold_supported_answer"
    CLOSED_BOOK = "closed_book_answer"
    UNSUPPORTED = "unsupported_answer"


def classify_evidence_state(
    *,
    evidence_available: bool,
    gold_found: int,
    gold_required: int,
    answer_correct: bool,
    non_gold_support: bool = False,
    closed_book_correct: bool = False,
) -> EvidenceState:
    if not evidence_available:
        return EvidenceState.EVIDENCE_UNAVAILABLE
    if gold_found <= 0 and not answer_correct:
        return EvidenceState.RETRIEVAL_FAILURE
    if answer_correct and non_gold_support:
        return EvidenceState.NON_GOLD_SUPPORTED
    if answer_correct and closed_book_correct:
        return EvidenceState.CLOSED_BOOK
    if gold_found < gold_required:
        return (
            EvidenceState.UNSUPPORTED
            if answer_correct
            else EvidenceState.INCOMPLETE_RETRIEVAL
        )
    if not answer_correct:
        return EvidenceState.EVIDENCE_USE_FAILURE
    return EvidenceState.COMPLETE_RETRIEVAL
