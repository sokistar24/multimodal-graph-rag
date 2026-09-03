"""
Generates CROSS-PAPER multi-hop questions for the SPIQA corpus.

This is the hypothesis-testing question set. Each question requires one fact
from each of TWO different papers that share an entity, so answering it fully
demands evidence that no single retrieved paper contains. This is the condition
under which knowledge-graph traversal could genuinely add evidence, unlike the
within-page multi-hop protocol used for PubLayNet.

Design constraints (each mirrors a mechanic of the runtime pipeline):
  * Candidate entities are taken from the TRIPLE STORE itself
    (build_triples(with_sources=True)), restricted to entities appearing in
    triples from >= 2 distinct papers. A null graph result on these questions
    is therefore attributable to the mechanism, not to questions the graph
    never covered.
  * Entities must be >= MIN_ENTITY_LEN chars — the same min_len=5 filter
    graph_facts_for_query applies at query time.
  * The shared entity MUST appear verbatim as a whole word in the question
    (validated with the same \\b regex the pipeline uses), otherwise the graph
    lookup cannot fire and the question silently tests nothing.
  * No modality cues (figure/table/chart/image/graph/diagram/plot), same as
    the other question sets.

Output items:
  {"q", "source", "source2", "answer", "type": "multihop_cross", "entity"}

"source" is the primary gold paper so compare_all.py's standard single-gold
metrics run unchanged; "source2" is the second gold paper, logged for the
both-papers-retrieved analysis (was the second paper in the text top-k? did
the provenance filter admit its graph facts?).

Usage (RAG_CORPUS need not be set; the corpus dir is passed explicitly):
    python generate_questions_spiqa_cross.py --n 50
    python generate_questions_spiqa_cross.py --n 5 --verbose
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

from multimodal_graph_rag.retrieval.graph import build_triples
from multimodal_graph_rag.retrieval.text import chunk_text, load_file

load_dotenv()


# Author: DeepSeek — the same author as the text/multi-hop sets, and OUTSIDE the
# four generator families under test. Authoring with gpt-4o-mini would let one
# of the evaluated generators write its own exam (the locked design forbids it).
def deepseek_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing from .env")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


client = deepseek_client()

GEN_MODEL = "deepseek-chat"
CORPUS_DIR = "spiqa_corpus"
OUT_FILE = os.path.join("data", "questions", "questions_spiqa_multihop_cross.json")
SEED = 42
MIN_ENTITY_LEN = 5  # MUST equal graph_facts_for_query's min_len
OVERGEN_FACTOR = 2.0  # generate ~2x, then skim manually

BANNED_WORDS = {
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

# Words that reference the two-source structure of the generation setup. The
# answering system sees one fused prompt — "first context" means nothing to it.
# Prompting alone did not stop these (the model copied a template literally),
# so they are hard-rejected here. Over-generation absorbs the loss of the rare
# legitimate use of "first"/"second".
STRUCTURAL_WORDS = {
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

# Generic CS entities that appear in half the corpus and make trivially vague
# questions. Extend after eyeballing a --verbose run.
STOPLIST = {
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


def whole_word(entity: str, text: str) -> bool:
    return (
        re.search(r"\b" + re.escape(entity.lower()) + r"\b", text.lower()) is not None
    )


def collect_candidates(corpus_dir):
    """
    Walk the triple store and return, per entity, the papers it occurs in and
    one representative (triple, chunk) per paper.

    Entity = a triple subject or object, lowercased, >= MIN_ENTITY_LEN chars,
    not in the stoplist. Exactly matches what graph_facts_for_query can match.
    """
    triples = build_triples(corpus_dir=corpus_dir, with_sources=True)

    # entity -> source -> one example triple
    occurrences = defaultdict(dict)
    for s, r, o, src in triples:
        for ent in (str(s).lower().strip(), str(o).lower().strip()):
            if len(ent) < MIN_ENTITY_LEN or ent in STOPLIST:
                continue
            if src not in occurrences[ent]:
                occurrences[ent][src] = (s, r, o)

    multi = {e: srcs for e, srcs in occurrences.items() if len(srcs) >= 2}
    print(f"Entities in the graph: {len(occurrences)}; in >= 2 papers: {len(multi)}")
    return multi


def chunk_containing(entity, source, corpus_dir):
    """Return the chunk of `source` that mentions the entity (longest match),
    so the question model sees real surrounding context, not just the triple."""
    text = load_file(os.path.join(corpus_dir, source))
    hits = [c for c in chunk_text(text) if whole_word(entity, c)]
    return max(hits, key=len) if hits else None


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

Excerpt A (from paper {src_a}):
{chunk_a}

Excerpt B (from paper {src_b}):
{chunk_b}
"""


def generate_pair_question(entity, src_a, chunk_a, src_b, chunk_b):
    prompt = PROMPT.format(
        entity=entity,
        src_a=src_a,
        src_b=src_b,
        chunk_a=chunk_a[:1500],
        chunk_b=chunk_b[:1500],
    )
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (
        resp.choices[0]
        .message.content.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )
    try:
        qa = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(qa, dict) or qa.get("q") in (None, "", "SKIP"):
        return None
    return qa


def validate(qa, entity):
    """Programmatic checks mirroring the pipeline's runtime behaviour, plus the
    canonical leak/orphan validators from generate_questions.py."""
    from generate_questions import has_orphan_reference, leaks_modality

    q = qa["q"]
    if leaks_modality(q):
        return "modality leak (canonical validator)"
    if has_orphan_reference(q):
        return "orphan reference (canonical validator)"
    if not whole_word(entity, q):
        return "entity not whole-word in question"
    q_words = set(re.findall(r"[a-z]+", q.lower()))
    hit = q_words & BANNED_WORDS
    if hit:
        return f"modality cue: {sorted(hit)}"
    hit = q_words & STRUCTURAL_WORDS
    if hit:
        return f"structural reference: {sorted(hit)}"
    if len(q) < 30:
        return "question too short/vague"
    if not qa.get("answer") or qa["answer"] == "SKIP":
        return "no answer"
    return None


def main(n_target, verbose=False):
    rng = random.Random(SEED)
    multi = collect_candidates(CORPUS_DIR)

    # Prefer entities in FEW papers (2-4): they are specific, not generic hubs.
    ranked = sorted(multi.items(), key=lambda kv: (len(kv[1]), -len(kv[0])))
    rng.shuffle(ranked[:0])  # keep deterministic order; shuffle only pair choice below

    n_attempts = int(n_target * OVERGEN_FACTOR)
    out, used_pairs = [], set()

    for entity, srcs in ranked:
        if len(out) >= n_attempts:
            break
        sources = sorted(srcs.keys())
        src_a, src_b = rng.sample(sources, 2)
        pair_key = frozenset((src_a, src_b))
        if pair_key in used_pairs:
            continue  # spread questions across paper pairs

        chunk_a = chunk_containing(entity, src_a, CORPUS_DIR)
        chunk_b = chunk_containing(entity, src_b, CORPUS_DIR)
        if not chunk_a or not chunk_b:
            continue

        qa = generate_pair_question(entity, src_a, chunk_a, src_b, chunk_b)
        if qa is None:
            continue
        problem = validate(qa, entity)
        if problem:
            if verbose:
                print(f"  [reject] {entity!r}: {problem}")
            continue

        used_pairs.add(pair_key)
        out.append(
            {
                "q": qa["q"],
                "source": src_a,  # primary gold (compare_all unchanged)
                "source2": src_b,  # second gold, for the coverage analysis
                "answer": qa["answer"],
                "type": "multihop_cross",
                "entity": entity,
            }
        )
        if verbose:
            print(f"  [{len(out)}] entity={entity!r}  {src_a} + {src_b}")
            print(f"       Q: {qa['q']}")
            print(f"       A: {qa['answer']}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(out)} candidate questions -> {OUT_FILE}")
    print(
        f"Manually skim down to the best {n_target}: remove duplicates in spirit, "
        f"questions answerable from one paper alone, and vague pairings."
    )
    kept_entities = [o["entity"] for o in out]
    print(f"Entities used: {len(set(kept_entities))} distinct")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Cross-paper multi-hop questions for SPIQA, seeded from the triple store."
    )
    ap.add_argument(
        "--n",
        type=int,
        default=50,
        help="target set size after manual skim (default 50; ~2x generated)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="print accepted/rejected candidates as they are produced",
    )
    args = ap.parse_args()
    main(args.n, verbose=args.verbose)
