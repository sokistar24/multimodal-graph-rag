"""Graph-seeded cross-paper questions (the in-graph mechanism set).

Each question needs one fact from each of two papers that share an entity in
the triple store, so no single retrieved paper contains the full evidence.
Because the linking entity is drawn from the same graph that ``+KGret`` later
traverses, this set is a controlled mechanism test and is labelled
``graph_seeded``; it cannot on its own support a general claim about
cross-document retrieval, which needs the independently authored set.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..clients import ModelClient
from ..corpora.loaders import chunk_text, load_file
from ..retrieval.graph import MIN_ENTITY_LENGTH, build_triples
from ..schemas import QuestionRecord, content_hash, write_questions
from .authoring import TEXT_AUTHOR
from .validators import (
    has_orphan_reference,
    leaks_modality,
    parse_json_object,
    whole_word,
)

CONSTRUCTION = "deepseek-graph-seeded-cross-paper"
DEFAULT_SEED = 42
OVERGENERATION_FACTOR = 2.0
MIN_QUESTION_CHARS = 30
BANNED_WORDS = frozenset(
    {
        "figure",
        "table",
        "chart",
        "image",
        "graph",
        "diagram",
        "plot",
        "caption",
        "photo",
        "picture",
    }
)
STRUCTURAL_WORDS = frozenset(
    {
        "context",
        "contexts",
        "excerpt",
        "excerpts",
        "passage",
        "passages",
        "paper",
        "papers",
        "source",
        "sources",
        "first",
        "second",
        "former",
        "latter",
        "respectively",
        "aforementioned",
    }
)
STOPLIST = frozenset(
    {
        "model",
        "models",
        "method",
        "methods",
        "approach",
        "dataset",
        "datasets",
        "results",
        "training",
        "network",
        "networks",
        "algorithm",
        "performance",
        "accuracy",
        "learning",
        "system",
        "table",
        "figure",
        "section",
        "paper",
        "authors",
        "baseline",
        "experiments",
        "function",
        "features",
        "value",
        "values",
    }
)

PROMPT = """You write evaluation questions for a document question-answering system.

You are given ONE shared entity and TWO excerpts, each from a DIFFERENT scientific paper.
Write ONE question whose complete answer requires ONE fact from EACH excerpt.

CRITICAL: the system answering the question does NOT know there are two sources.
The question must read as a single natural question. Each half must therefore be
anchored by a specific NAMED method, dataset, system, metric, or term that appears
in ITS OWN excerpt — never by ordering words.

GOOD (each half anchored by its own named referent, MTSA and TREC):
{{"q": "What is the role of MTSA in question-type classification, and how is question-type classification categorized in TREC?", "answer": "..."}}

BAD (ordinal references are meaningless to the answering system — never do this):
{{"q": "What is X in the first context, and what is Y in the second context?", "answer": "..."}}

Rules:
- The question MUST contain the shared entity verbatim: "{entity}"
- The answer must combine one fact from excerpt A and one fact from excerpt B.
- Each fact must be explicitly stated in its excerpt. No outside knowledge.
- NEVER use any of these words in the question: context, contexts, excerpt,
  excerpts, passage, passages, paper, papers, source, sources, first, second,
  former, latter, respectively, aforementioned.
- Do not use: figure, table, chart, image, graph, diagram, plot.
- If you cannot anchor BOTH halves with named referents from their own excerpts,
  respond with {{"q": "SKIP", "answer": "SKIP"}}.
- Respond with ONLY a JSON object: {{"q": "...", "answer": "..."}}

Shared entity: {entity}

Excerpt A (from paper {source_a}):
{chunk_a}

Excerpt B (from paper {source_b}):
{chunk_b}
"""


@dataclass(frozen=True)
class CrossPaperReport:
    written: int
    target: int
    entities_available: int
    rejected: dict[str, int]
    output: Path


def collect_candidates(
    corpus_dir: Path, cache_path: Path
) -> dict[str, dict[str, tuple[str, str, str]]]:
    """Entities appearing in triples from at least two papers, with one example each."""
    occurrences: dict[str, dict[str, tuple[str, str, str]]] = defaultdict(dict)
    for triple in build_triples(corpus_dir, cache_path):
        for entity in (triple.subject.lower().strip(), triple.obj.lower().strip()):
            if len(entity) < MIN_ENTITY_LENGTH or entity in STOPLIST:
                continue
            occurrences[entity].setdefault(
                triple.source, (triple.subject, triple.relation, triple.obj)
            )
    return {
        entity: papers for entity, papers in occurrences.items() if len(papers) >= 2
    }


def chunk_containing(entity: str, source: str, corpus_dir: Path) -> str | None:
    hits = [
        chunk
        for chunk in chunk_text(load_file(corpus_dir / source))
        if whole_word(entity, chunk)
    ]
    return max(hits, key=len) if hits else None


def validate(question: str, answer: str, entity: str) -> str | None:
    if leaks_modality(question):
        return "modality leak"
    if has_orphan_reference(question):
        return "orphan reference"
    if not whole_word(entity, question):
        return "entity not whole-word in question"
    words = set(re.findall(r"[a-z]+", question.lower()))
    if words & BANNED_WORDS:
        return "modality cue"
    if words & STRUCTURAL_WORDS:
        return "structural reference"
    if len(question) < MIN_QUESTION_CHARS:
        return "too short"
    if not answer or answer == "SKIP":
        return "no answer"
    return None


def author_cross_paper(
    client: ModelClient,
    *,
    corpus_dir: str | Path,
    cache_path: str | Path,
    target: int,
    out_path: str | Path,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
) -> CrossPaperReport:
    corpus = Path(corpus_dir)
    rng = random.Random(seed)
    candidates = collect_candidates(corpus, Path(cache_path))
    ranked = sorted(candidates.items(), key=lambda kv: (len(kv[1]), -len(kv[0])))
    attempts_wanted = int(target * OVERGENERATION_FACTOR)
    records: list[QuestionRecord] = []
    used_pairs: set[frozenset[str]] = set()
    rejected: dict[str, int] = {}

    for entity, papers in ranked:
        if len(records) >= attempts_wanted:
            break
        source_a, source_b = rng.sample(sorted(papers), 2)
        pair = frozenset((source_a, source_b))
        if pair in used_pairs:
            continue
        chunk_a = chunk_containing(entity, source_a, corpus)
        chunk_b = chunk_containing(entity, source_b, corpus)
        if not chunk_a or not chunk_b:
            continue
        prompt = PROMPT.format(
            entity=entity,
            source_a=source_a,
            source_b=source_b,
            chunk_a=chunk_a[:1500],
            chunk_b=chunk_b[:1500],
        )
        reply = parse_json_object(
            client.call(TEXT_AUTHOR, user=prompt, max_tokens=400).text
        )
        if reply is None or reply.get("q") in (None, "", "SKIP"):
            rejected["author declined or unparseable"] = (
                rejected.get("author declined or unparseable", 0) + 1
            )
            continue
        question = str(reply["q"]).strip()
        answer = str(reply.get("answer", "")).strip()
        problem = validate(question, answer, entity)
        if problem:
            rejected[problem] = rejected.get(problem, 0) + 1
            if verbose:
                print(f"  [reject] {entity!r}: {problem}")
            continue
        used_pairs.add(pair)
        records.append(
            QuestionRecord(
                id=content_hash(
                    {"q": question, "answer": answer, "sources": [source_a, source_b]}
                )[:16],
                question=question,
                answer=answer,
                gold_sources=(source_a, source_b),
                question_type="multihop_cross",
                construction_method=CONSTRUCTION,
                evidence_requirement="text-multi-source",
                graph_seeded=True,
                metadata={"entity": entity},
            )
        )
        if verbose:
            print(
                f"  [{len(records)}] {entity!r}: {source_a} + {source_b}\n       Q: {question}"
            )

    output = Path(out_path)
    write_questions(records, output, source_file=str(cache_path))
    return CrossPaperReport(len(records), target, len(candidates), rejected, output)
