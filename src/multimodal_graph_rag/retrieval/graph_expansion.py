"""Retrieval-stage graph expansion (the ``+KGret`` arm).

Entities matched in the question identify graph-connected documents. Chunks
from those documents that dense retrieval missed are added to the candidate set:
the standard top-``k`` dense passages plus up to ``n_bridge`` passages from
bridge documents, ranked by embedding similarity to the query. This is the
channel the generation-stage ``+KG`` system structurally excludes, because its
provenance filter restricts graph facts to documents dense retrieval already
returned.

The retrieval-only A/B in :func:`bridge_report` measures the pre-registered
primary outcome without any generation: the fraction of questions whose full
gold provenance is in the candidate set before and after expansion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from ..clients import ModelClient
from ..schemas import QuestionRecord
from .graph import KnowledgeGraph
from .text import Passage, TextIndex

DEFAULT_BASE_K = 3
DEFAULT_BRIDGE_SLOTS = 2
DEEP_SEARCH_K = 300
MAX_NEIGHBOURS = 20


def bridge_pages(
    query: str,
    graph: KnowledgeGraph,
    *,
    hops: int = 1,
    max_neighbours: int = MAX_NEIGHBOURS,
) -> set[str]:
    """Documents reachable from entities that appear in the question.

    ``hops=1`` covers documents of edges incident to matched nodes, which is
    sufficient when the linking entity appears verbatim in the question.
    ``hops=2`` additionally covers documents of edges incident to those nodes'
    neighbours, which HotpotQA-style bridge questions require because the
    bridge entity is never named. ``max_neighbours`` caps hub blow-up.
    """
    if hops not in (1, 2):
        raise ValueError("hops must be 1 or 2")
    matched = graph.match_entities(query)
    pages: set[str] = set()
    for node in matched:
        pages |= graph.pages_of(node)
    if hops == 2:
        for node in matched:
            for neighbour in sorted(graph.neighbours(node))[:max_neighbours]:
                pages |= graph.pages_of(neighbour)
    return pages


@dataclass(frozen=True)
class BridgedRetrieval:
    base: tuple[Passage, ...]
    bridged: tuple[Passage, ...]
    candidate_pages: frozenset[str]

    @property
    def ranked_sources(self) -> list[str]:
        return [p.source for p in self.base] + [p.source for p in self.bridged]


def retrieve_with_bridge(
    query: str,
    index: TextIndex,
    graph: KnowledgeGraph,
    client: ModelClient,
    *,
    k: int = DEFAULT_BASE_K,
    n_bridge: int = DEFAULT_BRIDGE_SLOTS,
    hops: int = 1,
    deep_k: int = DEEP_SEARCH_K,
) -> BridgedRetrieval:
    """Dense top-``k`` plus up to ``n_bridge`` passages from bridge documents."""
    embedded = client.embed([query], model=index.metadata.embedding_model)
    query_vector = embedded.vectors[0]
    base = index.search_vector(query_vector, k)
    base_sources = {passage.source for passage in base}
    candidates = bridge_pages(query, graph, hops=hops) - base_sources
    if not candidates or n_bridge < 1:
        return BridgedRetrieval(tuple(base), (), frozenset(candidates))

    bridged: list[Passage] = []
    used_pages: set[str] = set()
    for passage in index.search_vector(query_vector, deep_k):
        if passage.source in candidates and passage.source not in used_pages:
            bridged.append(
                Passage(
                    source=passage.source,
                    text=passage.text,
                    score=passage.score,
                    rank=len(base) + len(bridged) + 1,
                )
            )
            used_pages.add(passage.source)
            if len(bridged) >= n_bridge:
                break
    return BridgedRetrieval(tuple(base), tuple(bridged), frozenset(candidates))


@dataclass(frozen=True)
class BridgeOutcome:
    question_id: str
    question: str
    gold_sources: tuple[str, ...]
    base_sources: tuple[str, ...]
    bridged_sources: tuple[tuple[str, float], ...]
    candidate_count: int
    missing_gold_status: dict[str, str]
    complete_before: bool
    complete_after: bool
    rescued: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BridgeReport:
    outcomes: tuple[BridgeOutcome, ...]

    @property
    def count(self) -> int:
        return len(self.outcomes)

    def summary(self) -> dict[str, int]:
        return {
            "questions": self.count,
            "any_gold_before": sum(
                any(g in o.base_sources for g in o.gold_sources) for o in self.outcomes
            ),
            "any_gold_after": sum(
                any(
                    g in o.base_sources or g in {s for s, _ in o.bridged_sources}
                    for g in o.gold_sources
                )
                for o in self.outcomes
            ),
            "complete_before": sum(o.complete_before for o in self.outcomes),
            "complete_after": sum(o.complete_after for o in self.outcomes),
            "bridge_fired": sum(bool(o.bridged_sources) for o in self.outcomes),
            "rescued": sum(o.rescued for o in self.outcomes),
            "slot_lost": sum(
                status == "slot_lost"
                for o in self.outcomes
                for status in o.missing_gold_status.values()
            ),
            "not_candidate": sum(
                status == "not_candidate"
                for o in self.outcomes
                for status in o.missing_gold_status.values()
            ),
        }


def bridge_report(
    questions: Sequence[QuestionRecord],
    index: TextIndex,
    graph: KnowledgeGraph,
    client: ModelClient,
    *,
    k: int = DEFAULT_BASE_K,
    n_bridge: int = DEFAULT_BRIDGE_SLOTS,
    hops: int = 1,
) -> BridgeReport:
    """Retrieval-only A/B: gold completeness before and after graph expansion."""
    outcomes = []
    for question in questions:
        result = retrieve_with_bridge(
            question.question, index, graph, client, k=k, n_bridge=n_bridge, hops=hops
        )
        base_sources = {p.source for p in result.base}
        all_sources = base_sources | {p.source for p in result.bridged}
        golds = question.text_gold_sources or question.gold_sources
        before = [g in base_sources for g in golds]
        after = [g in all_sources for g in golds]
        missing = [g for g in golds if g not in all_sources]
        outcomes.append(
            BridgeOutcome(
                question_id=question.id,
                question=question.question,
                gold_sources=tuple(golds),
                base_sources=tuple(sorted(base_sources)),
                bridged_sources=tuple(
                    (p.source, round(p.score, 4)) for p in result.bridged
                ),
                candidate_count=len(result.candidate_pages),
                missing_gold_status={
                    g: ("slot_lost" if g in result.candidate_pages else "not_candidate")
                    for g in missing
                },
                complete_before=all(before),
                complete_after=all(after),
                rescued=all(after) and not all(before),
            )
        )
    return BridgeReport(tuple(outcomes))
