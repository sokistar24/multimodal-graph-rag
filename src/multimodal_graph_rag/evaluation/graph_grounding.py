"""Automated grounding audit of every extracted triple.

The knowledge graph is built by asking a model to read each chunk and emit
``[subject, relation, object]`` triples. Nothing in that pipeline checks that
the entities it emits actually occur in the text it read, so a graph of tens of
thousands of triples can carry hallucinated endpoints and misattributed
provenance without any of it being visible.

This checks, for every triple in the cache, whether its subject and object
appear in the chunk the triple was extracted from. It runs over the whole graph
rather than a sample, costs nothing, and needs no annotator.

What it establishes and what it does not
    It establishes *lexical grounding*: the endpoints are present in the source
    text, so the triple is anchored to the passage it claims to come from, and
    its provenance edge points at a chunk that really contains those entities.
    A low rate is strong evidence of extraction error.

    It does NOT establish that the triple is true, that the relation is
    correct, or that the relation is specific enough to be useful. A model can
    connect two entities that both appear in a chunk with a relation the chunk
    does not support, and this check will pass it. Relation correctness is a
    judgement, and this audit deliberately does not claim to make it.

Report the rates as triple grounding, never as triple precision.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..corpora.loaders import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    iter_corpus_chunks,
)
from ..errors import CacheError
from ..retrieval.graph import TripleStore, chunk_key, normalise_entity

WHITESPACE = re.compile(r"\s+")


def _searchable(text: str) -> str:
    return WHITESPACE.sub(" ", text.lower())


@dataclass(frozen=True)
class GroundingReport:
    """Counts over every triple in the cache for one corpus."""

    corpus: str
    chunks: int
    chunks_without_triples: int
    triples: int
    subject_present: int
    object_present: int
    both_present: int
    neither_present: int
    relation_present: int
    distinct_subjects: int
    distinct_objects: int
    numeric_endpoints: int

    def _rate(self, count: int) -> float:
        return count / self.triples if self.triples else 0.0

    @property
    def grounding_rate(self) -> float:
        """Fraction of triples whose subject and object both occur in the source."""
        return self._rate(self.both_present)

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "chunks": self.chunks,
            "chunks_without_triples": self.chunks_without_triples,
            "triples": self.triples,
            "subject_present": self.subject_present,
            "object_present": self.object_present,
            "both_present": self.both_present,
            "neither_present": self.neither_present,
            "relation_present": self.relation_present,
            "distinct_subjects": self.distinct_subjects,
            "distinct_objects": self.distinct_objects,
            "numeric_endpoints": self.numeric_endpoints,
            "grounding_rate": round(self.grounding_rate, 4),
            "subject_rate": round(self._rate(self.subject_present), 4),
            "object_rate": round(self._rate(self.object_present), 4),
            "relation_rate": round(self._rate(self.relation_present), 4),
        }


def audit_triple_grounding(
    corpus_dir: str | Path,
    cache_path: str | Path,
    *,
    corpus: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> GroundingReport:
    """Check every cached triple against the chunk it was extracted from.

    The corpus must be chunked exactly as it was when the triples were
    extracted, since the cache is keyed by chunk hash. A chunk with no cache
    entry means the cache does not match this corpus, which is an error rather
    than something to skip.
    """
    store = TripleStore.load(cache_path)
    if not store.entries:
        raise CacheError(f"triple cache is empty: {cache_path}")

    chunks = triples = 0
    empty_chunks = 0
    subject_present = object_present = both_present = neither_present = 0
    relation_present = numeric_endpoints = 0
    subjects: set[str] = set()
    objects: set[str] = set()

    for chunk in iter_corpus_chunks(corpus_dir, chunk_size, overlap):
        chunks += 1
        key = chunk_key(chunk.text)
        if key not in store:
            raise CacheError(
                f"chunk from {chunk.source} has no cached triples in {cache_path}; "
                "the cache was built from a different corpus or chunking"
            )
        entries = store.entries[key]
        if not entries:
            empty_chunks += 1
            continue
        haystack = _searchable(chunk.text)
        for triple in entries:
            if len(triple) != 3:
                raise CacheError(f"malformed triple in {cache_path}: {triple!r}")
            subject, relation, obj = (normalise_entity(str(part)) for part in triple)
            triples += 1
            subjects.add(subject)
            objects.add(obj)
            has_subject = bool(subject) and subject in haystack
            has_object = bool(obj) and obj in haystack
            subject_present += has_subject
            object_present += has_object
            both_present += has_subject and has_object
            neither_present += not has_subject and not has_object
            relation_present += bool(relation) and relation in haystack
            numeric_endpoints += any(
                part.replace(".", "", 1).replace(",", "").isdigit()
                for part in (subject, obj)
                if part
            )

    return GroundingReport(
        corpus=corpus or Path(corpus_dir).name,
        chunks=chunks,
        chunks_without_triples=empty_chunks,
        triples=triples,
        subject_present=subject_present,
        object_present=object_present,
        both_present=both_present,
        neither_present=neither_present,
        relation_present=relation_present,
        distinct_subjects=len(subjects),
        distinct_objects=len(objects),
        numeric_endpoints=numeric_endpoints,
    )


def canonicalisation_collisions(
    entries: Mapping[str, list[list[str]]], minimum: int = 2
) -> dict[str, list[str]]:
    """Surface forms that collapse to the same node once normalised.

    Every distinct surface form becomes one node, so forms differing only by
    case or surrounding whitespace merge silently. Listing them shows how much
    the graph relies on that accident, which is the canonicalisation quality
    the audit protocol asks about.
    """
    forms: dict[str, set[str]] = {}
    for triples in entries.values():
        for triple in triples:
            if len(triple) != 3:
                continue
            for raw in (str(triple[0]), str(triple[2])):
                forms.setdefault(normalise_entity(raw), set()).add(raw)
    return {
        node: sorted(variants)
        for node, variants in sorted(forms.items())
        if len(variants) >= minimum
    }
