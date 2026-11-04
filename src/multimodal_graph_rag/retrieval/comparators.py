"""Published graph-retrieval comparators served from precomputed retrievals.

The comparator (HippoRAG 2 first) runs in its own environment because it
brings its own model stack. It never runs inside an evaluation. Instead a
one-off pass in that environment indexes our corpus chunks, retrieves for
every question in a set, maps each returned chunk text back to the chunk's
source file, and writes one JSON file. The evaluation pipeline then serves
those rankings as ordinary passages through :class:`PrecomputedRetrievals`,
so the comparator differs from the other arms in retrieval only.

The file records the corpus fingerprint and the question-set digest it was
built for, and loading refuses a corpus that has changed since.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..corpora.loaders import iter_corpus_chunks
from ..errors import ConfigurationError, MissingEvidenceError
from ..schemas import QuestionRecord, content_hash, load_questions
from .text import Passage, corpus_fingerprint

SCHEMA_VERSION = 1


def _question_digest(question_file: str | Path) -> str:
    return content_hash([q.to_dict() for q in load_questions(question_file)])


def write_retrievals(
    path: str | Path,
    *,
    retrievals: Mapping[str, Sequence[Mapping[str, Any]]],
    corpus_dir: str | Path,
    question_file: str | Path,
    comparator: str,
    num_to_retrieve: int,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write ranked retrievals keyed by question id, with their provenance."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "comparator": comparator,
        "corpus_dir": str(corpus_dir),
        "corpus_fingerprint": corpus_fingerprint(corpus_dir),
        "question_file": str(question_file),
        "questions_sha256": _question_digest(question_file),
        "num_to_retrieve": num_to_retrieve,
        "metadata": dict(metadata or {}),
        "retrievals": {
            question_id: [
                {
                    "source": str(item["source"]),
                    "text": str(item.get("text", "")),
                    "score": float(item.get("score", 0.0)),
                }
                for item in ranked
            ]
            for question_id, ranked in retrievals.items()
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return destination


@dataclass(frozen=True)
class PrecomputedRetrievals:
    """Ranked passages per question, produced outside the evaluation."""

    comparator: str
    corpus_fingerprint: str
    questions_sha256: str
    num_to_retrieve: int
    metadata: Mapping[str, Any]
    retrievals: Mapping[str, tuple[Passage, ...]]
    path: Path

    @classmethod
    def load(
        cls, path: str | Path, *, corpus_dir: str | Path | None = None
    ) -> PrecomputedRetrievals:
        source = Path(path)
        if not source.is_file():
            raise ConfigurationError(f"comparator retrievals file not found: {source}")
        with source.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ConfigurationError(
                f"comparator retrievals {source} use an unsupported schema version"
            )
        if corpus_dir is not None:
            actual = corpus_fingerprint(corpus_dir)
            if actual != raw.get("corpus_fingerprint"):
                raise ConfigurationError(
                    f"comparator retrievals {source} were built for a corpus with "
                    f"fingerprint {raw.get('corpus_fingerprint')}, but {corpus_dir} "
                    f"now has fingerprint {actual}; rebuild them"
                )
        retrievals = {
            question_id: tuple(
                Passage(
                    source=item["source"],
                    text=item.get("text", ""),
                    score=float(item.get("score", 0.0)),
                    rank=rank,
                )
                for rank, item in enumerate(ranked, 1)
            )
            for question_id, ranked in raw.get("retrievals", {}).items()
        }
        return cls(
            comparator=str(raw.get("comparator", "")),
            corpus_fingerprint=str(raw.get("corpus_fingerprint", "")),
            questions_sha256=str(raw.get("questions_sha256", "")),
            num_to_retrieve=int(raw.get("num_to_retrieve", 0)),
            metadata=dict(raw.get("metadata", {})),
            retrievals=retrievals,
            path=source,
        )

    def retrieve(self, question_id: str, k: int) -> tuple[Passage, ...]:
        try:
            ranked = self.retrievals[question_id]
        except KeyError as exc:
            raise MissingEvidenceError(
                f"{self.comparator} has no precomputed retrieval for question "
                f"{question_id!r} in {self.path}"
            ) from exc
        return ranked[:k]


class RetrievalEngine(Protocol):
    """The slice of ``hipporag.HippoRAG`` the adapter relies on."""

    def index(self, docs: list[str]) -> Any: ...

    def retrieve(self, queries: list[str], num_to_retrieve: int) -> Sequence[Any]: ...

    def get_graph_info(self) -> Mapping[str, Any]: ...


def precompute_hipporag(
    engine: RetrievalEngine,
    *,
    corpus_dir: str | Path,
    question_file: str | Path,
    output: str | Path,
    num_to_retrieve: int,
    llm_model: str,
    embedding_model: str,
    package_version: str = "",
) -> Path:
    """Index our chunks with the comparator and freeze its rankings to a file.

    The engine is fed the same chunks the dense baseline embeds, so every
    returned text maps back to a chunk source and the gold-provenance scoring
    works unchanged. A returned text that is not one of our chunks is an error
    rather than a silently dropped result.
    """
    chunks = list(iter_corpus_chunks(corpus_dir))
    source_of = {chunk.text: chunk.source for chunk in chunks}
    engine.index([chunk.text for chunk in chunks])
    questions = load_questions(question_file)
    solutions = engine.retrieve(
        [q.question for q in questions], num_to_retrieve=num_to_retrieve
    )
    retrievals: dict[str, list[dict[str, Any]]] = {}
    for question, solution in zip(questions, solutions, strict=True):
        ranked = []
        for text, score in zip(solution.docs, solution.doc_scores, strict=True):
            try:
                source = source_of[text]
            except KeyError as exc:
                raise MissingEvidenceError(
                    f"comparator returned a text for question {question.id!r} that "
                    "cannot be attributed to any corpus chunk"
                ) from exc
            ranked.append({"source": source, "text": text, "score": float(score)})
        retrievals[question.id] = ranked
    return write_retrievals(
        output,
        retrievals=retrievals,
        corpus_dir=corpus_dir,
        question_file=question_file,
        comparator="hipporag2",
        num_to_retrieve=num_to_retrieve,
        metadata={
            "package_version": package_version,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "chunk_count": len(chunks),
            "graph_info": dict(engine.get_graph_info()),
        },
    )


def comparator_report(
    questions: Sequence[QuestionRecord], retrievals: PrecomputedRetrievals, *, k: int
) -> dict[str, Any]:
    """Retrieval-only completeness of the comparator at a candidate budget."""
    outcomes = []
    complete = 0
    for question in questions:
        sources = [p.source for p in retrievals.retrieve(question.id, k)]
        golds = question.text_gold_sources or question.gold_sources
        missing = [g for g in golds if g not in sources]
        complete += not missing
        outcomes.append(
            {
                "question_id": question.id,
                "gold_sources": list(golds),
                "sources": sources,
                "missing_gold": missing,
                "complete": not missing,
            }
        )
    return {
        "summary": {"questions": len(questions), "complete": complete, "k": k},
        "outcomes": outcomes,
    }
