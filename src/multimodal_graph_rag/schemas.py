"""Versioned records shared by question generation, evaluation, and artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

SCHEMA_VERSION = 1
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QuestionRecord:
    id: str
    question: str
    answer: str
    gold_sources: tuple[str, ...]
    question_type: str
    construction_method: str = "legacy_unspecified"
    evidence_requirement: str = "text"
    graph_seeded: bool = False
    leakage_status: str = "not_audited"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.question.strip() or not self.answer.strip():
            raise ValueError("question id, question, and answer must be non-empty")
        if not self.gold_sources:
            raise ValueError(f"question {self.id!r} has no gold source")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported question schema version: {self.schema_version}"
            )

    @property
    def is_visual(self) -> bool:
        """True when any gold source is an image, so retrieval is scored on images."""
        return any(
            source.lower().endswith(IMAGE_SUFFIXES) for source in self.gold_sources
        )

    @property
    def text_gold_sources(self) -> tuple[str, ...]:
        return tuple(s for s in self.gold_sources if s.lower().endswith(".txt"))

    @classmethod
    def from_legacy(cls, item: Mapping[str, Any], index: int = 0) -> QuestionRecord:
        """Build a record from the ``{"q", "source", "answer", "type"}`` layout."""
        raw_sources = item.get("source", ())
        sources = [raw_sources] if isinstance(raw_sources, str) else list(raw_sources)
        if item.get("source2"):
            sources.append(str(item["source2"]))
        qtype = str(item.get("type", "text"))
        stable_id = (
            item.get("id")
            or content_hash(
                {"q": item.get("q"), "answer": item.get("answer"), "sources": sources}
            )[:16]
        )
        return cls(
            id=str(stable_id),
            question=str(item.get("q", "")),
            answer=str(item.get("answer", "")),
            gold_sources=tuple(str(s) for s in sources),
            question_type=qtype,
            construction_method=str(
                item.get("construction_method", "legacy_unspecified")
            ),
            evidence_requirement=str(
                item.get(
                    "evidence_requirement", "visual" if qtype == "figure" else "text"
                )
            ),
            graph_seeded=bool(
                item.get("graph_seeded", "cross" in qtype and "entity" in item)
            ),
            leakage_status=str(item.get("leakage_status", "not_audited")),
            metadata={
                k: v
                for k, v in item.items()
                if k not in {"q", "answer", "source", "source2"}
            },
        )

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> QuestionRecord:
        """Build a record from the versioned layout written by ``to_dict``."""
        data = dict(item)
        data["gold_sources"] = tuple(data["gold_sources"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gold_sources"] = list(self.gold_sources)
        return data


def load_questions(path: str | Path) -> list[QuestionRecord]:
    """Load a question file in either the legacy list or the versioned layout."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigurationError(f"question file does not exist: {file_path}")
    with file_path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if isinstance(raw, list):
        records = [
            QuestionRecord.from_legacy(item, index) for index, item in enumerate(raw)
        ]
    elif isinstance(raw, dict) and isinstance(raw.get("questions"), list):
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ConfigurationError(
                f"question file {file_path} uses an unsupported schema version"
            )
        records = [QuestionRecord.from_dict(item) for item in raw["questions"]]
    else:
        raise ConfigurationError(f"question file {file_path} has an unknown layout")
    if not records:
        raise ConfigurationError(f"question file {file_path} is empty")
    ids = [record.id for record in records]
    if len(set(ids)) != len(ids):
        raise ConfigurationError(f"question file {file_path} has duplicate ids")
    return records


def write_questions(
    records: Sequence[QuestionRecord], path: str | Path, *, source_file: str = ""
) -> str:
    """Write records in the versioned layout and return the content hash."""
    payload_records = [record.to_dict() for record in records]
    digest = content_hash(payload_records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_file": source_file,
        "question_count": len(payload_records),
        "questions_sha256": digest,
        "questions": payload_records,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return digest


@dataclass(frozen=True)
class EvidenceItem:
    modality: str
    source: str
    rank: int
    score: float | None = None
    content: str | None = None
    token_count: int | None = None
    selection_reason: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in {"text", "graph", "image"}:
            raise ValueError(f"unsupported modality: {self.modality}")
        if self.rank < 1:
            raise ValueError("evidence rank must be at least 1")


@dataclass(frozen=True)
class EvidenceBundle:
    items: tuple[EvidenceItem, ...] = ()
    candidate_budget: int = 0

    def sources(self, modality: str | None = None) -> tuple[str, ...]:
        return tuple(
            item.source
            for item in self.items
            if modality is None or item.modality == modality
        )

    def is_complete(self, gold_sources: Sequence[str]) -> bool:
        observed = set(self.sources())
        return all(source in observed for source in gold_sources)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config_hash: str
    question_id: str
    system: str
    answer: str
    evidence: EvidenceBundle
    metrics: Mapping[str, float | int | None]
    raw_judges: Mapping[str, Any] = field(default_factory=dict)
    human_label: Mapping[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    model_versions: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(self.to_dict()) + "\n")
