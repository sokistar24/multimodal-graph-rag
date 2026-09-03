"""The systems under comparison and the evidence-control arms.

Every runner receives the same :class:`RunContext` (retrieval components are
built once and shared) and returns a :class:`SystemOutput` that records exactly
which evidence reached the generator. Retrieval is identical across generators,
so any difference between runs traces to the generator alone.

    baseline      dense top-``k`` text passages
    +KG           baseline passages plus graph facts restricted to those passages
    +KGret        dense top-3 passages plus graph-expanded passages (equal budget)
    +multimodal   baseline passages plus a CLIP-retrieved crop (pixels or caption)
    +both         +KG and +multimodal together

    control:closed-book    no context at all
    control:shuffled       unrelated passages of matched length
    control:oracle         passages from every gold document
    control:partial-gold   passages from the first gold document only
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..clients import GenResult, ModelClient
from ..errors import ConfigurationError, MissingEvidenceError
from ..retrieval.graph import KnowledgeGraph
from ..retrieval.graph_expansion import retrieve_with_bridge
from ..retrieval.text import Passage, TextIndex
from ..retrieval.visual import ClipImageIndex, ImageHit
from ..schemas import EvidenceBundle, EvidenceItem, QuestionRecord
from . import EvidenceControl

IMAGES_RETRIEVE = 5
IMAGES_CAPTION = 3
IMAGES_PIXELS = 1

BASELINE_SYSTEM_PROMPT = (
    "Answer using only the provided context. "
    "If the answer isn't in it, say you don't know."
)
GRAPH_SYSTEM_PROMPT = (
    "Answer using only the provided context and knowledge-graph facts. "
    "If the answer isn't there, say you don't know."
)
PIXEL_SYSTEM_PROMPT = (
    "Answer using only the provided text context and the attached image(s). "
    "If the answer isn't there, say you don't know."
)
CAPTION_SYSTEM_PROMPT = (
    "Answer using only the provided text context and image descriptions. "
    "If the answer isn't there, say you don't know."
)
FULL_PIXEL_SYSTEM_PROMPT = (
    "Answer using only the provided text context, knowledge-graph facts, and "
    "the attached image(s). If the answer isn't in any of them, say you don't know."
)
FULL_CAPTION_SYSTEM_PROMPT = (
    "Answer using only the provided text context, knowledge-graph facts, and "
    "image descriptions. If the answer isn't in any of them, say you don't know."
)
CLOSED_BOOK_SYSTEM_PROMPT = (
    "Answer the question from your own knowledge. "
    "If you do not know the answer, say you don't know."
)


@dataclass(frozen=True)
class RunContext:
    client: ModelClient
    model: str
    text_index: TextIndex
    candidate_budget: int
    graph: KnowledgeGraph | None = None
    image_index: ClipImageIndex | None = None
    image_dir: Path | None = None
    bridge_base_k: int = 3
    bridge_hops: int = 1
    seed: int = 0
    images_retrieve: int = IMAGES_RETRIEVE
    images_caption: int = IMAGES_CAPTION
    images_pixels: int = IMAGES_PIXELS

    @property
    def bridge_slots(self) -> int:
        return self.candidate_budget - self.bridge_base_k

    def require_graph(self, system: str) -> KnowledgeGraph:
        if self.graph is None:
            raise MissingEvidenceError(f"{system} requires a knowledge graph")
        return self.graph

    def require_images(self, system: str) -> tuple[ClipImageIndex, Path]:
        if self.image_index is None or self.image_dir is None:
            raise MissingEvidenceError(f"{system} requires an image index")
        return self.image_index, self.image_dir


@dataclass(frozen=True)
class SystemOutput:
    result: GenResult
    text_passages: tuple[Passage, ...] = ()
    bridged_passages: tuple[Passage, ...] = ()
    graph_facts: tuple[str, ...] = ()
    images_scored: tuple[ImageHit, ...] = ()
    images_sent: tuple[ImageHit, ...] = ()
    image_paths: tuple[Path, ...] = ()
    context: str = ""
    selection_reason: str = "dense"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ranked_text_sources(self) -> list[str]:
        return [p.source for p in self.text_passages] + [
            p.source for p in self.bridged_passages
        ]

    @property
    def ranked_image_sources(self) -> list[str]:
        return [hit.name for hit in self.images_scored]

    def evidence_bundle(self, candidate_budget: int) -> EvidenceBundle:
        items: list[EvidenceItem] = []
        for passage in self.text_passages:
            items.append(
                EvidenceItem(
                    modality="text",
                    source=passage.source,
                    rank=passage.rank,
                    score=passage.score,
                    content=passage.text,
                    selection_reason=self.selection_reason,
                )
            )
        for passage in self.bridged_passages:
            items.append(
                EvidenceItem(
                    modality="text",
                    source=passage.source,
                    rank=passage.rank,
                    score=passage.score,
                    content=passage.text,
                    selection_reason="graph-bridged",
                )
            )
        for rank, fact in enumerate(self.graph_facts, 1):
            items.append(
                EvidenceItem(
                    modality="graph",
                    source="knowledge-graph",
                    rank=rank,
                    content=fact,
                    selection_reason="entity-match within retrieved sources",
                )
            )
        sent = {hit.name for hit in self.images_sent}
        for hit in self.images_scored:
            items.append(
                EvidenceItem(
                    modality="image",
                    source=hit.name,
                    rank=hit.rank,
                    score=hit.score,
                    content=hit.caption,
                    selection_reason="clip"
                    + (" (sent)" if hit.name in sent else " (scored only)"),
                )
            )
        return EvidenceBundle(items=tuple(items), candidate_budget=candidate_budget)


def format_passages(passages: Sequence[Passage], label: str = "") -> str:
    suffix = f" | {label}" if label else ""
    return "\n\n".join(f"[{p.source}{suffix}] {p.text}" for p in passages)


def format_facts(facts: Sequence[str]) -> str:
    return "\n".join(f"- {fact}" for fact in facts) if facts else "(none)"


def format_captions(hits: Sequence[ImageHit]) -> str:
    return (
        "\n".join(f"- ({hit.name}) {hit.caption}" for hit in hits) if hits else "(none)"
    )


def judge_context(*parts: str) -> str:
    """The evidence exactly as the faithfulness judge should see it."""
    return "\n".join(part for part in parts if part)


def _generate(
    ctx: RunContext, system_prompt: str, user: str, images: Sequence[Path] = ()
) -> GenResult:
    return ctx.client.call(ctx.model, system=system_prompt, user=user, images=images)


def _select_images(
    ctx: RunContext, question: QuestionRecord, use_pixels: bool, system: str
) -> tuple[tuple[ImageHit, ...], tuple[ImageHit, ...], tuple[Path, ...]]:
    index, image_dir = ctx.require_images(system)
    scored = tuple(index.retrieve(question.question, ctx.images_retrieve))
    if use_pixels:
        if not ctx.client.spec(ctx.model).vision:
            raise ConfigurationError(
                f"{ctx.model} cannot run the pixel condition for {system}"
            )
        sent = scored[: ctx.images_pixels]
        paths = tuple(image_dir / hit.name for hit in sent)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise MissingEvidenceError(f"retrieved crops do not exist: {missing}")
        return scored, sent, paths
    return scored, scored[: ctx.images_caption], ()


def run_baseline(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    passages = tuple(
        ctx.text_index.retrieve(question.question, ctx.client, ctx.candidate_budget)
    )
    user = f"Context:\n{format_passages(passages)}\n\nQuestion: {question.question}"
    return SystemOutput(
        result=_generate(ctx, BASELINE_SYSTEM_PROMPT, user),
        text_passages=passages,
        context=judge_context(*(p.text for p in passages)),
    )


def run_kg(question: QuestionRecord, ctx: RunContext, use_pixels: bool) -> SystemOutput:
    graph = ctx.require_graph("+KG")
    passages = tuple(
        ctx.text_index.retrieve(question.question, ctx.client, ctx.candidate_budget)
    )
    facts = tuple(
        graph.facts_for_query(
            question.question, allowed_sources=[p.source for p in passages]
        )
    )
    user = (
        f"Context passages:\n{format_passages(passages)}\n\n"
        f"Knowledge-graph facts:\n{format_facts(facts)}\n\n"
        f"Question: {question.question}"
    )
    return SystemOutput(
        result=_generate(ctx, GRAPH_SYSTEM_PROMPT, user),
        text_passages=passages,
        graph_facts=facts,
        context=judge_context(*(p.text for p in passages), *facts),
    )


def run_kgret(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    graph = ctx.require_graph("+KGret")
    retrieval = retrieve_with_bridge(
        question.question,
        ctx.text_index,
        graph,
        ctx.client,
        k=ctx.bridge_base_k,
        n_bridge=ctx.bridge_slots,
        hops=ctx.bridge_hops,
    )
    context = format_passages(retrieval.base)
    if retrieval.bridged:
        context += "\n\n" + format_passages(retrieval.bridged, "graph-bridged")
    user = f"Context:\n{context}\n\nQuestion: {question.question}"
    return SystemOutput(
        result=_generate(ctx, BASELINE_SYSTEM_PROMPT, user),
        text_passages=retrieval.base,
        bridged_passages=retrieval.bridged,
        context=judge_context(
            *(p.text for p in retrieval.base),
            *(f"[graph-bridged] {p.text}" for p in retrieval.bridged),
        ),
        metadata={"candidate_pages": len(retrieval.candidate_pages)},
    )


def run_multimodal(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    passages = tuple(
        ctx.text_index.retrieve(question.question, ctx.client, ctx.candidate_budget)
    )
    scored, sent, paths = _select_images(ctx, question, use_pixels, "+multimodal")
    text_block = f"Text context:\n{format_passages(passages)}\n\n"
    if paths:
        user = f"{text_block}Question: {question.question}"
        result = _generate(ctx, PIXEL_SYSTEM_PROMPT, user, paths)
    else:
        user = (
            f"{text_block}Relevant images (described):\n{format_captions(sent)}\n\n"
            f"Question: {question.question}"
        )
        result = _generate(ctx, CAPTION_SYSTEM_PROMPT, user)
    return SystemOutput(
        result=result,
        text_passages=passages,
        images_scored=scored,
        images_sent=sent,
        image_paths=paths,
        context=judge_context(
            *(p.text for p in passages), *(hit.caption for hit in sent)
        ),
    )


def run_both(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    graph = ctx.require_graph("+both")
    passages = tuple(
        ctx.text_index.retrieve(question.question, ctx.client, ctx.candidate_budget)
    )
    facts = tuple(
        graph.facts_for_query(
            question.question, allowed_sources=[p.source for p in passages]
        )
    )
    scored, sent, paths = _select_images(ctx, question, use_pixels, "+both")
    head = (
        f"Text context:\n{format_passages(passages)}\n\n"
        f"Knowledge-graph facts:\n{format_facts(facts)}\n\n"
    )
    if paths:
        result = _generate(
            ctx, FULL_PIXEL_SYSTEM_PROMPT, f"{head}Question: {question.question}", paths
        )
    else:
        user = (
            f"{head}Relevant images (described):\n{format_captions(sent)}\n\n"
            f"Question: {question.question}"
        )
        result = _generate(ctx, FULL_CAPTION_SYSTEM_PROMPT, user)
    return SystemOutput(
        result=result,
        text_passages=passages,
        graph_facts=facts,
        images_scored=scored,
        images_sent=sent,
        image_paths=paths,
        context=judge_context(
            *(p.text for p in passages), *facts, *(hit.caption for hit in sent)
        ),
    )


def _passages_from_sources(
    ctx: RunContext, question: QuestionRecord, sources: Sequence[str], reason: str
) -> tuple[Passage, ...]:
    """Chunks of the given documents ranked by similarity to the question."""
    wanted = set(sources)
    if not wanted & ctx.text_index.source_names:
        raise MissingEvidenceError(
            f"none of {sorted(wanted)} exist in the corpus for question {question.id}"
        )
    ranked = ctx.text_index.retrieve(question.question, ctx.client, ctx.text_index.size)
    chosen: list[Passage] = []
    for passage in ranked:
        if passage.source in wanted:
            chosen.append(
                Passage(passage.source, passage.text, passage.score, len(chosen) + 1)
            )
            if len(chosen) == ctx.candidate_budget:
                break
    return tuple(chosen)


def run_closed_book(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    return SystemOutput(
        result=_generate(
            ctx, CLOSED_BOOK_SYSTEM_PROMPT, f"Question: {question.question}"
        ),
        selection_reason="closed-book",
    )


def run_shuffled(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    """Unrelated passages of matched total length, drawn from non-gold documents.

    Gold documents are excluded outright, and so are the specific passages the
    normal retriever returned, so the control carries the same context length
    with none of the evidence.
    """
    normal = ctx.text_index.retrieve(
        question.question, ctx.client, ctx.candidate_budget
    )
    target_chars = sum(len(p.text) for p in normal)
    gold = set(question.gold_sources)
    retrieved_chunks = {p.text for p in normal}
    pool = [
        position
        for position, source in enumerate(ctx.text_index.sources)
        if source not in gold
        and ctx.text_index.chunks[position] not in retrieved_chunks
    ]
    if not pool:
        raise MissingEvidenceError(
            f"no non-gold passages available for the shuffled control on "
            f"question {question.id}"
        )
    rng = random.Random(f"{ctx.seed}:{question.id}")
    rng.shuffle(pool)
    chosen: list[Passage] = []
    total = 0
    for position in pool:
        chosen.append(
            Passage(
                ctx.text_index.sources[position],
                ctx.text_index.chunks[position],
                0.0,
                len(chosen) + 1,
            )
        )
        total += len(chosen[-1].text)
        if total >= target_chars or len(chosen) == ctx.candidate_budget:
            break
    passages = tuple(chosen)
    user = f"Context:\n{format_passages(passages)}\n\nQuestion: {question.question}"
    return SystemOutput(
        result=_generate(ctx, BASELINE_SYSTEM_PROMPT, user),
        text_passages=passages,
        context=judge_context(*(p.text for p in passages)),
        selection_reason="shuffled-unrelated",
        metadata={"matched_chars": target_chars},
    )


def run_oracle(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    sources = question.text_gold_sources or question.gold_sources
    passages = _passages_from_sources(ctx, question, sources, "oracle")
    user = f"Context:\n{format_passages(passages)}\n\nQuestion: {question.question}"
    return SystemOutput(
        result=_generate(ctx, BASELINE_SYSTEM_PROMPT, user),
        text_passages=passages,
        context=judge_context(*(p.text for p in passages)),
        selection_reason="oracle-gold",
    )


def run_partial_gold(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    sources = question.text_gold_sources or question.gold_sources
    passages = _passages_from_sources(ctx, question, sources[:1], "partial-gold")
    user = f"Context:\n{format_passages(passages)}\n\nQuestion: {question.question}"
    return SystemOutput(
        result=_generate(ctx, BASELINE_SYSTEM_PROMPT, user),
        text_passages=passages,
        context=judge_context(*(p.text for p in passages)),
        selection_reason="partial-gold",
    )


SystemRunner = Callable[[QuestionRecord, RunContext, bool], SystemOutput]

SYSTEMS: dict[str, SystemRunner] = {
    "baseline": run_baseline,
    "+KG": run_kg,
    "+KGret": run_kgret,
    "+multimodal": run_multimodal,
    "+both": run_both,
    EvidenceControl.CLOSED_BOOK.system_name: run_closed_book,
    EvidenceControl.SHUFFLED.system_name: run_shuffled,
    EvidenceControl.ORACLE.system_name: run_oracle,
    EvidenceControl.PARTIAL_GOLD.system_name: run_partial_gold,
}
GRAPH_SYSTEMS = frozenset({"+KG", "+KGret", "+both"})
IMAGE_SYSTEMS = frozenset({"+multimodal", "+both"})
DEFAULT_SYSTEMS: tuple[str, ...] = ("baseline", "+KG", "+multimodal", "+both")


def resolve_systems(names: Sequence[str]) -> list[str]:
    """Validate system names and return them in registry order."""
    unknown = [name for name in names if name not in SYSTEMS]
    if unknown:
        raise ConfigurationError(
            f"unknown system(s): {unknown}. Choices: {list(SYSTEMS)}"
        )
    wanted = set(names)
    return [name for name in SYSTEMS if name in wanted]
