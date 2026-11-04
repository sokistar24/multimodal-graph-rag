"""The systems under comparison and the evidence-control arms.

Every runner receives the same :class:`RunContext` (retrieval components are
built once and shared) and returns a :class:`SystemOutput` that records exactly
which evidence reached the generator. Retrieval is identical across generators,
so any difference between runs traces to the generator alone.

    dense-top3    dense top-3 text passages
    baseline      dense top-``k`` text passages (``k`` is the candidate budget)
    bm25          BM25 top-``k`` over the same chunks
    hybrid-rrf    dense and BM25 rankings fused by reciprocal rank, top-``k``
    +KG           dense top-3 passages plus graph facts restricted to them
    +KGret        dense top-3 plus graph-expanded passages (equal budget)
    +lexical      dense top-3 plus chunks naming a question entity, no edges
    +random       dense top-3 plus random chunks from unretrieved sources
    +HippoRAG     a published graph retriever (HippoRAG 2) at the same budget,
                  served from retrievals precomputed in its own environment
    +multimodal   baseline passages plus a CLIP-retrieved crop (pixels or caption)
    +both         +multimodal plus graph facts restricted to its passages

    control:closed-book    no context at all
    control:shuffled       unrelated passages of matched length
    control:oracle         passages from every gold document
    control:partial-gold   passages from the first gold document only
    control:oracle-image   +multimodal with the gold crop in place of CLIP's

Every passage-only arm generates through :func:`answer_from_passages`, so the
arms differ in which passages they retrieve and in nothing else the generator
sees; passages carry no marker of the arm that chose them. :func:`replay`
rebuilds an arm's prompt from a stored evidence bundle to generate again.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..clients import GENERATOR_MAX_TOKENS, GenResult, ModelClient
from ..errors import ConfigurationError, MissingEvidenceError
from ..retrieval.baselines import (
    BM25Index,
    bm25_passages,
    hybrid_passages,
    lexical_entity_expansion,
    random_padding,
)
from ..retrieval.comparators import PrecomputedRetrievals
from ..retrieval.graph import KnowledgeGraph
from ..retrieval.graph_expansion import retrieve_with_bridge
from ..retrieval.text import Passage, TextIndex
from ..retrieval.visual import IMAGE_SUFFIXES, ClipImageIndex, ImageHit
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
    comparator: PrecomputedRetrievals | None = None
    image_index: ClipImageIndex | None = None
    image_dir: Path | None = None
    lexicon: BM25Index | None = None
    bridge_base_k: int = 3
    bridge_hops: int = 1
    seed: int = 0
    images_retrieve: int = IMAGES_RETRIEVE
    images_caption: int = IMAGES_CAPTION
    images_pixels: int = IMAGES_PIXELS
    max_tokens: int = GENERATOR_MAX_TOKENS

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

    def require_comparator(self, system: str) -> PrecomputedRetrievals:
        if self.comparator is None:
            raise ConfigurationError(
                f"{system} requires precomputed comparator retrievals "
                "(set comparator_retrievals in the config)"
            )
        return self.comparator

    def require_lexicon(self, system: str) -> BM25Index:
        if self.lexicon is None:
            raise ConfigurationError(f"{system} requires a BM25 index over the chunks")
        return self.lexicon

    def query_vector(self, question: QuestionRecord) -> np.ndarray:
        embedded = self.client.embed(
            [question.question], model=self.text_index.metadata.embedding_model
        )
        return embedded.vectors[0]


@dataclass(frozen=True)
class SystemOutput:
    result: GenResult
    text_passages: tuple[Passage, ...] = ()
    # Passages an expansion arm added after the dense top-k.
    bridged_passages: tuple[Passage, ...] = ()
    graph_facts: tuple[str, ...] = ()
    images_scored: tuple[ImageHit, ...] = ()
    images_sent: tuple[ImageHit, ...] = ()
    image_paths: tuple[Path, ...] = ()
    context: str = ""
    selection_reason: str = "dense"
    expansion_reason: str = "graph-bridged"
    image_reason: str = "clip"
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
                    selection_reason=self.expansion_reason,
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
                    selection_reason=self.image_reason
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
    return ctx.client.call(
        ctx.model,
        system=system_prompt,
        user=user,
        images=images,
        max_tokens=ctx.max_tokens,
    )


def answer_from_passages(
    ctx: RunContext,
    question: QuestionRecord,
    passages: Sequence[Passage],
    added: Sequence[Passage] = (),
    *,
    reason: str,
    added_reason: str = "",
    metadata: dict[str, object] | None = None,
) -> SystemOutput:
    """Generate from passages alone, with the prompt every passage arm shares."""
    shown = (*passages, *added)
    user = f"Context:\n{format_passages(shown)}\n\nQuestion: {question.question}"
    return SystemOutput(
        result=_generate(ctx, BASELINE_SYSTEM_PROMPT, user),
        text_passages=tuple(passages),
        bridged_passages=tuple(added),
        context=judge_context(*(p.text for p in shown)),
        selection_reason=reason,
        expansion_reason=added_reason or reason,
        metadata=metadata or {},
    )


def _dense(ctx: RunContext, question: QuestionRecord, k: int) -> tuple[Passage, ...]:
    return tuple(ctx.text_index.retrieve(question.question, ctx.client, k))


def _with_images(
    ctx: RunContext,
    question: QuestionRecord,
    passages: Sequence[Passage],
    facts: Sequence[str] | None,
    scored: tuple[ImageHit, ...],
    sent: tuple[ImageHit, ...],
    paths: tuple[Path, ...],
) -> SystemOutput:
    """Generate from passages, images or captions, and graph facts unless None."""
    head = f"Text context:\n{format_passages(passages)}\n\n"
    if facts is not None:
        head += f"Knowledge-graph facts:\n{format_facts(facts)}\n\n"
    if paths:
        prompt = PIXEL_SYSTEM_PROMPT if facts is None else FULL_PIXEL_SYSTEM_PROMPT
        result = _generate(ctx, prompt, f"{head}Question: {question.question}", paths)
    else:
        prompt = CAPTION_SYSTEM_PROMPT if facts is None else FULL_CAPTION_SYSTEM_PROMPT
        user = (
            f"{head}Relevant images (described):\n{format_captions(sent)}\n\n"
            f"Question: {question.question}"
        )
        result = _generate(ctx, prompt, user)
    return SystemOutput(
        result=result,
        text_passages=tuple(passages),
        graph_facts=tuple(facts or ()),
        images_scored=scored,
        images_sent=sent,
        image_paths=paths,
        context=judge_context(
            *(p.text for p in passages), *(facts or ()), *(hit.caption for hit in sent)
        ),
    )


def _select_images(
    ctx: RunContext, question: QuestionRecord, use_pixels: bool, system: str
) -> tuple[tuple[ImageHit, ...], tuple[ImageHit, ...], tuple[Path, ...]]:
    index, image_dir = ctx.require_images(system)
    scored = tuple(index.retrieve(question.question, ctx.images_retrieve))
    return _deliver(ctx, scored, image_dir, use_pixels, system, ctx.images_caption)


def _deliver(
    ctx: RunContext,
    scored: tuple[ImageHit, ...],
    image_dir: Path,
    use_pixels: bool,
    system: str,
    captions: int,
) -> tuple[tuple[ImageHit, ...], tuple[ImageHit, ...], tuple[Path, ...]]:
    """The hits to send: top crops as pixels, or the top captions as text."""
    if use_pixels:
        if not ctx.client.spec(ctx.model).vision:
            raise ConfigurationError(
                f"{ctx.model} cannot run the pixel condition for {system}"
            )
        sent = scored[: ctx.images_pixels]
        paths = tuple(image_dir / hit.name for hit in sent)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise MissingEvidenceError(f"image crops do not exist: {missing}")
        return scored, sent, paths
    return scored, scored[:captions], ()


@dataclass(frozen=True)
class Retrieved:
    """The passages one passage arm selects for a question, before generation."""

    passages: tuple[Passage, ...]
    added: tuple[Passage, ...] = ()
    reason: str = "dense"
    added_reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        return [p.source for p in (*self.passages, *self.added)]


def retrieve_dense_top3(question: QuestionRecord, ctx: RunContext) -> Retrieved:
    return Retrieved(_dense(ctx, question, ctx.bridge_base_k))


def retrieve_dense(question: QuestionRecord, ctx: RunContext) -> Retrieved:
    return Retrieved(_dense(ctx, question, ctx.candidate_budget))


def retrieve_bm25(question: QuestionRecord, ctx: RunContext) -> Retrieved:
    passages = bm25_passages(
        ctx.text_index,
        ctx.require_lexicon("bm25"),
        question.question,
        ctx.candidate_budget,
    )
    return Retrieved(passages, reason="bm25")


def retrieve_hybrid(question: QuestionRecord, ctx: RunContext) -> Retrieved:
    passages = hybrid_passages(
        ctx.text_index,
        ctx.require_lexicon("hybrid-rrf"),
        question.question,
        ctx.query_vector(question),
        ctx.candidate_budget,
    )
    return Retrieved(passages, reason="hybrid-rrf")


def retrieve_kgret(question: QuestionRecord, ctx: RunContext) -> Retrieved:
    retrieval = retrieve_with_bridge(
        question.question,
        ctx.text_index,
        ctx.require_graph("+KGret"),
        ctx.client,
        k=ctx.bridge_base_k,
        n_bridge=ctx.bridge_slots,
        hops=ctx.bridge_hops,
    )
    return Retrieved(
        retrieval.base,
        retrieval.bridged,
        added_reason="graph-bridged",
        metadata={"candidate_pages": len(retrieval.candidate_pages)},
    )


def retrieve_lexical(question: QuestionRecord, ctx: RunContext) -> Retrieved:
    retrieval = lexical_entity_expansion(
        ctx.text_index,
        ctx.require_graph("+lexical"),
        question.question,
        ctx.query_vector(question),
        k=ctx.bridge_base_k,
        n_added=ctx.bridge_slots,
    )
    return Retrieved(
        retrieval.base, retrieval.added, added_reason="entity-string match"
    )


def retrieve_random(question: QuestionRecord, ctx: RunContext) -> Retrieved:
    retrieval = random_padding(
        ctx.text_index,
        ctx.query_vector(question),
        k=ctx.bridge_base_k,
        n_added=ctx.bridge_slots,
        seed=f"{ctx.seed}:{question.id}:random-padding",
    )
    return Retrieved(retrieval.base, retrieval.added, added_reason="random padding")


def retrieve_hipporag(question: QuestionRecord, ctx: RunContext) -> Retrieved:
    """The published comparator at the baseline budget."""
    passages = ctx.require_comparator("+HippoRAG").retrieve(
        question.id, ctx.candidate_budget
    )
    return Retrieved(passages, reason="hipporag")


PASSAGE_RETRIEVERS: dict[str, Callable[[QuestionRecord, RunContext], Retrieved]] = {
    "dense-top3": retrieve_dense_top3,
    "baseline": retrieve_dense,
    "bm25": retrieve_bm25,
    "hybrid-rrf": retrieve_hybrid,
    "+KGret": retrieve_kgret,
    "+lexical": retrieve_lexical,
    "+random": retrieve_random,
    "+HippoRAG": retrieve_hipporag,
}


def _answer(
    ctx: RunContext, question: QuestionRecord, retrieved: Retrieved
) -> SystemOutput:
    return answer_from_passages(
        ctx,
        question,
        retrieved.passages,
        retrieved.added,
        reason=retrieved.reason,
        added_reason=retrieved.added_reason,
        metadata=retrieved.metadata,
    )


def run_dense_top3(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    return _answer(ctx, question, retrieve_dense_top3(question, ctx))


def run_baseline(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    return _answer(ctx, question, retrieve_dense(question, ctx))


def run_bm25(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    return _answer(ctx, question, retrieve_bm25(question, ctx))


def run_hybrid(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    return _answer(ctx, question, retrieve_hybrid(question, ctx))


def run_kg(question: QuestionRecord, ctx: RunContext, use_pixels: bool) -> SystemOutput:
    """Dense top-3 plus facts from those passages; dense top-3 is its control."""
    graph = ctx.require_graph("+KG")
    passages = _dense(ctx, question, ctx.bridge_base_k)
    facts = tuple(
        graph.facts_for_query(
            question.question, allowed_sources=[p.source for p in passages]
        )
    )
    return answer_with_facts(ctx, question, passages, facts)


def answer_with_facts(
    ctx: RunContext,
    question: QuestionRecord,
    passages: tuple[Passage, ...],
    facts: tuple[str, ...],
) -> SystemOutput:
    """Generate from passages and graph facts, with the +KG prompt."""
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
    return _answer(ctx, question, retrieve_kgret(question, ctx))


def run_lexical(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    return _answer(ctx, question, retrieve_lexical(question, ctx))


def run_random(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    return _answer(ctx, question, retrieve_random(question, ctx))


def run_hipporag(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    """The published comparator at the baseline budget, same prompt as baseline."""
    return _answer(ctx, question, retrieve_hipporag(question, ctx))


def passage_arm_report(
    questions: Sequence[QuestionRecord], ctx: RunContext, arms: Sequence[str]
) -> dict[str, object]:
    """Gold completeness of each passage arm, before anything is generated.

    It calls the same retrieval functions the generation runs use, so what it
    reports is exactly what each arm would hand the generator.
    """
    totals = {arm: {"any_gold": 0, "complete": 0, "passages": 0} for arm in arms}
    outcomes = []
    for question in questions:
        golds = question.text_gold_sources or question.gold_sources
        row: dict[str, object] = {
            "question_id": question.id,
            "gold_sources": list(golds),
        }
        for arm in arms:
            sources = PASSAGE_RETRIEVERS[arm](question, ctx).sources
            found = [gold in sources for gold in golds]
            totals[arm]["any_gold"] += any(found)
            totals[arm]["complete"] += all(found)
            totals[arm]["passages"] += len(sources)
            row[arm] = {"sources": sources, "complete": all(found)}
        outcomes.append(row)
    count = max(len(questions), 1)
    summary = {
        arm: {
            "questions": len(questions),
            "any_gold": total["any_gold"],
            "complete": total["complete"],
            "mean_passages": round(total["passages"] / count, 2),
        }
        for arm, total in totals.items()
    }
    return {"summary": summary, "outcomes": outcomes}


def run_multimodal(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    passages = _dense(ctx, question, ctx.candidate_budget)
    scored, sent, paths = _select_images(ctx, question, use_pixels, "+multimodal")
    return _with_images(ctx, question, passages, None, scored, sent, paths)


def run_both(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    """+multimodal plus graph facts, so the two differ only in the facts."""
    graph = ctx.require_graph("+both")
    passages = _dense(ctx, question, ctx.candidate_budget)
    facts = tuple(
        graph.facts_for_query(
            question.question, allowed_sources=[p.source for p in passages]
        )
    )
    scored, sent, paths = _select_images(ctx, question, use_pixels, "+both")
    return _with_images(ctx, question, passages, facts, scored, sent, paths)


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
    normal = _dense(ctx, question, ctx.candidate_budget)
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
        chosen.append(ctx.text_index.passage_at(position, 0.0, len(chosen) + 1))
        total += len(chosen[-1].text)
        if total >= target_chars or len(chosen) == ctx.candidate_budget:
            break
    return answer_from_passages(
        ctx,
        question,
        chosen,
        reason="shuffled-unrelated",
        metadata={"matched_chars": target_chars},
    )


def run_oracle(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    sources = question.text_gold_sources or question.gold_sources
    passages = _passages_from_sources(ctx, question, sources, "oracle")
    return answer_from_passages(ctx, question, passages, reason="oracle-gold")


def run_partial_gold(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    sources = question.text_gold_sources or question.gold_sources
    passages = _passages_from_sources(ctx, question, sources[:1], "partial-gold")
    return answer_from_passages(ctx, question, passages, reason="partial-gold")


def run_oracle_image(
    question: QuestionRecord, ctx: RunContext, use_pixels: bool
) -> SystemOutput:
    """+multimodal with the gold crop supplied in place of CLIP's ranking.

    The text passages are the baseline's, so the gap to +multimodal is image
    retrieval alone, and what this arm still gets wrong is a failure to read
    the figure rather than to find it.
    """
    system = EvidenceControl.ORACLE_IMAGE.system_name
    index, image_dir = ctx.require_images(system)
    gold = [s for s in question.gold_sources if s.lower().endswith(IMAGE_SUFFIXES)]
    if not gold:
        raise MissingEvidenceError(
            f"question {question.id} has no gold image for the oracle-image arm"
        )
    unindexed = [name for name in gold if name not in index.captions]
    if unindexed:
        raise MissingEvidenceError(
            f"gold crops of question {question.id} are not in the image index: "
            f"{unindexed}"
        )
    hits = tuple(
        ImageHit(name=name, caption=index.captions[name], score=1.0, rank=rank)
        for rank, name in enumerate(gold, 1)
    )
    scored, sent, paths = _deliver(
        ctx, hits, image_dir, use_pixels, system, captions=len(hits)
    )
    passages = _dense(ctx, question, ctx.candidate_budget)
    output = _with_images(ctx, question, passages, None, scored, sent, paths)
    return replace(output, image_reason="oracle-gold")


SystemRunner = Callable[[QuestionRecord, RunContext, bool], SystemOutput]

SYSTEMS: dict[str, SystemRunner] = {
    "dense-top3": run_dense_top3,
    "baseline": run_baseline,
    "bm25": run_bm25,
    "hybrid-rrf": run_hybrid,
    "+KG": run_kg,
    "+KGret": run_kgret,
    "+lexical": run_lexical,
    "+random": run_random,
    "+HippoRAG": run_hipporag,
    "+multimodal": run_multimodal,
    "+both": run_both,
    EvidenceControl.CLOSED_BOOK.system_name: run_closed_book,
    EvidenceControl.SHUFFLED.system_name: run_shuffled,
    EvidenceControl.ORACLE.system_name: run_oracle,
    EvidenceControl.PARTIAL_GOLD.system_name: run_partial_gold,
    EvidenceControl.ORACLE_IMAGE.system_name: run_oracle_image,
}
GRAPH_SYSTEMS = frozenset({"+KG", "+KGret", "+lexical", "+both"})
LEXICON_SYSTEMS = frozenset({"bm25", "hybrid-rrf"})
COMPARATOR_SYSTEMS = frozenset({"+HippoRAG"})
IMAGE_SYSTEMS = frozenset(
    {"+multimodal", "+both", EvidenceControl.ORACLE_IMAGE.system_name}
)
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


def replay(
    system: str,
    question: QuestionRecord,
    ctx: RunContext,
    evidence: Mapping[str, Any],
) -> SystemOutput:
    """Generate again from a stored evidence bundle, without retrieving.

    The prompt is rebuilt from the passages and facts the bundle holds, in
    their stored order, so the new answer sees exactly the evidence the stored
    one did. Arms that sent images are not replayed.
    """
    if system not in SYSTEMS:
        raise ConfigurationError(f"unknown system {system!r}")
    items = evidence["items"]
    if system in IMAGE_SYSTEMS or any(item["modality"] == "image" for item in items):
        raise ConfigurationError(f"{system} sent images; rerun it instead of replaying")
    if system == EvidenceControl.CLOSED_BOOK.system_name:
        return run_closed_book(question, ctx, use_pixels=False)
    text = [item for item in items if item["modality"] == "text"]
    if not text:
        raise ConfigurationError(f"the stored {system} evidence has no passages")
    passages = tuple(
        Passage(item["source"], item["content"], item["score"], item["rank"])
        for item in text
    )
    if system == "+KG":
        facts = tuple(item["content"] for item in items if item["modality"] == "graph")
        return answer_with_facts(ctx, question, passages, facts)
    # Base passages come first; an expansion arm's follow under another reason.
    reason = text[0]["selection_reason"]
    split = next(
        (i for i, item in enumerate(text) if item["selection_reason"] != reason),
        len(text),
    )
    return answer_from_passages(
        ctx,
        question,
        passages[:split],
        passages[split:],
        reason=reason,
        added_reason=text[split]["selection_reason"] if split < len(text) else "",
    )
