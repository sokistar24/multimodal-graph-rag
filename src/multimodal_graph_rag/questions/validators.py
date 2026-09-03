"""Programmatic checks that every authored question must pass.

modality leak     the question reveals where the answer lives ("figure")
orphan reference  a bare "the study"/"the participants" with no anchoring
                  entity, so no single document can be identified
container/source  "this work", "the passage", or a page or arXiv identifier
over-length       two welded questions rather than one
"""

from __future__ import annotations

import json
import re

LEAK_PATTERN = re.compile(
    r"\b(figure|fig\.?|chart|image|graph|table|diagram|plot|panel|"
    r"shown|depicted|illustrat|pictured|above|below|visual)\b",
    re.I,
)
ORPHAN_PATTERN = re.compile(
    r"\b(?:the|this|these|those|that)\s+"
    r"(?:[\w-]+\s+){0,2}"
    r"(stud(?:y|ies)|paper|article|workshop|trial|survey|experiment|"
    r"interviews?|questionnaires?|participants?|patients?|subjects?|"
    r"cohort|sample|authors?|researchers?|analysis|research|"
    r"investigation|project|programme|program|intervention|"
    r"dataset|data\s?set|model|method|approach|framework|"
    r"work|review|report|findings?|results?)\b",
    re.I,
)
ANCHOR_PATTERN = re.compile(
    r"[A-Z][a-z]{2,}"
    r"|\b\d{4}\b"
    r"|\b\d+(?:\.\d+)?\s?"
    r"(?:%|mg|ml|kg|mm|cm|µg|nm|mmol|units?|years?|months?|weeks?|days?)\b"
)
CONTAINER_PATTERN = re.compile(
    r"\b(?:this|the|that)\s+(?:present\s+|current\s+)?"
    r"(?:passage|work|paper|text|manuscript|article|excerpt)\b"
    r"|\baccording to the (?:passage|text|authors?)\b"
    r"|\b(?:the|this)\s+proposed\s+(?:loss|method|model|approach|framework|"
    r"algorithm|system|technique|mechanism|architecture|network)\b",
    re.I,
)
SOURCE_ID_PATTERN = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b|\bPMC\d+\b", re.I)
QUESTION_WORD_PATTERN = re.compile(r"^\s*(what|how|which|when|where|who|why)\b", re.I)
MAX_QUESTION_CHARS = 220


def leaks_modality(question: str) -> bool:
    return bool(LEAK_PATTERN.search(question))


def has_orphan_reference(question: str) -> bool:
    """True when a bare definite reference has no anchoring entity, year, or unit."""
    if not ORPHAN_PATTERN.search(question):
        return False
    body = QUESTION_WORD_PATTERN.sub("", question)
    return not ANCHOR_PATTERN.search(body)


def bad_reference(question: str) -> bool:
    return bool(
        CONTAINER_PATTERN.search(question) or SOURCE_ID_PATTERN.search(question)
    )


def rejection_reason(question: str) -> str | None:
    """Why a question fails the protocol, or ``None`` if it passes."""
    if bad_reference(question):
        return "container reference or source identifier"
    if len(question) > MAX_QUESTION_CHARS:
        return "over-length"
    if has_orphan_reference(question):
        return "orphan reference"
    if leaks_modality(question):
        return "modality leak"
    return None


def parse_json_array(text: str) -> list | None:
    """Parse a JSON array, tolerating the markdown fences models add."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def parse_json_object(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def whole_word(entity: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(entity.lower())}\b", text.lower()) is not None
