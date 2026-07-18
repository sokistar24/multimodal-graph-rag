"""
Knowledge-graph enhanced RAG, built on top of rag_basics.

Three stages:
  1. Extract : gpt-4o-mini reads each chunk and returns (subject, relation, object)
               triples. These are cached to triples_cache.json so extraction runs
               once rather than on every run.
  2. Graph   : the triples are loaded into a directed NetworkX graph.
  3. ask_graph: at query time, graph nodes mentioned in the question are matched,
               their neighbouring triples are pulled and rendered as text, and those
               facts are added to the prompt alongside the normal retrieved chunks
               (late fusion of text and structured knowledge).

The baseline (rag_basics) is unchanged; ask_graph is the +KG variant. The graph
augments generation (it adds facts to the prompt); it does not change retrieval.

Adds one dependency beyond rag_basics:
    pip install networkx
"""
import os
import re
import json
import glob
import hashlib
import networkx as nx
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from rag_basics import (
    load_file, chunk_text, embed, retrieve,
    build_index, CORPUS_DIR, CHAT_MODEL,
)
from explainability import format_explanation, text_evidence, parse_level

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EXTRACT_MODEL = "gpt-4o-mini"

# One cache per corpus. A single shared file would append HotpotQA's triples to
# PubLayNet's and build one graph containing both — silently, with no error.
#
# Keyed on the directory NAME, not a content fingerprint (unlike the index and
# CLIP caches). Extraction costs hours and real money, and chunk-level hashing
# already handles staleness: a changed chunk misses the cache and is re-extracted
# on its own. Invalidating everything because one file's mtime moved would be
# expensive and pointless.
CACHE_FILE = "triples_cache.json"        # legacy path, migrated on first use
# The legacy cache was built from PubLayNet. Naming it explicitly stops the
# migration handing those triples to a different corpus.
LEGACY_CACHE_CORPUS = "publaynet_corpus"


def _cache_path(corpus_dir):
    """Cache filename for a corpus, e.g. triples_cache_publaynet_corpus.json."""
    name = os.path.basename(os.path.normpath(corpus_dir))
    return f"triples_cache_{name}.json"


def _migrate_legacy_cache(corpus_dir):
    """
    Move the old shared triples_cache.json to its corpus-specific name.

    The legacy file holds PubLayNet's 7,307 extracted chunks — hours of work and
    real API spend. Renaming rather than ignoring it means the switch to
    per-corpus caches costs nothing.

    Guarded two ways:
      • only migrates into LEGACY_CACHE_CORPUS, the corpus the old file actually
        came from. Otherwise pointing at HotpotQA would rename PubLayNet's cache
        into HotpotQA's slot and hand it a graph of biomedical triples.
      • only when the target doesn't already exist, so it is safe to re-run.
    """
    target = _cache_path(corpus_dir)
    if os.path.exists(target) or not os.path.exists(CACHE_FILE):
        return target
    if os.path.basename(os.path.normpath(corpus_dir)) != LEGACY_CACHE_CORPUS:
        # A different corpus: leave the legacy file alone and start fresh.
        return target
    try:
        os.rename(CACHE_FILE, target)
        print(f"  migrated {CACHE_FILE} -> {target} (existing triples preserved)")
    except OSError as e:
        print(f"  could not migrate {CACHE_FILE} ({e}); using it as-is")
        return CACHE_FILE
    return target


# ---------- 1. extract triples (cached) ----------
def _chunk_key(chunk):
    """Stable id for a chunk, used as its cache key."""
    return hashlib.md5(chunk.encode("utf-8")).hexdigest()

def extract_triples(chunk):
    """Asks the model for (subject, relation, object) triples from one chunk."""
    prompt = (
        "Extract the key facts from the TEXT as a list of triples. "
        "Each triple is [subject, relation, object], capturing one relationship. "
        "Use short noun phrases for subject and object. "
        "Respond with ONLY a JSON list, no markdown, e.g. "
        '[["Augustus","was","first Roman emperor"], ["aqueducts","carried","water"]]\n\n'
        f"TEXT:\n{chunk}"
    )
    resp = client.chat.completions.create(
        model=EXTRACT_MODEL, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        triples = json.loads(raw)
        return [t for t in triples if isinstance(t, list) and len(t) == 3]   # keep well-formed only
    except json.JSONDecodeError:
        return []

def build_triples(corpus_dir=CORPUS_DIR, with_sources=False):
    """Extracts triples for every chunk in the corpus, using the cache where possible.

    with_sources=False -> [[s, r, o], ...]                    (original behaviour)
    with_sources=True  -> [(s, r, o, source_file), ...]

    Provenance matters: without it, graph_facts_for_query can only match entity
    names across the WHOLE corpus, so a common node like "patient" drags in facts
    from hundreds of unrelated papers. Carrying the source page lets the graph be
    restricted to the pages retrieval actually returned.

    The cache is keyed by chunk hash, so provenance is recovered by re-walking the
    corpus — no re-extraction, no new API cost.
    """
    cache_file = _migrate_legacy_cache(corpus_dir)
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)

    # gather every chunk first so the total is known up front.
    # Keep the source file alongside each chunk — this is what the original
    # version discarded.
    chunks_to_process = []          # [(chunk_text, source_basename)]
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*"))):
        text = load_file(path)
        src = os.path.basename(path)
        for chunk in chunk_text(text):
            chunks_to_process.append((chunk, src))

    total = len(chunks_to_process)
    n_cached = sum(1 for c, _ in chunks_to_process if _chunk_key(c) in cache)
    n_todo = total - n_cached

    if n_todo > 0:
        print(f"Building knowledge graph: {n_todo} chunks need triple extraction "
              f"({n_cached} already cached). Runs once, then caches.")
    else:
        print(f"Knowledge graph: all {total} chunks cached, loading instantly.")

    all_triples = []
    new_extractions = 0
    # progress milestones at 20/40/60/80/100% of the remaining work
    milestones = {int(n_todo * p / 100) for p in (20, 40, 60, 80, 100)} if n_todo else set()

    for chunk, src in tqdm(chunks_to_process, desc="Extracting KG triples"):
        key = _chunk_key(chunk)
        if key not in cache:
            cache[key] = extract_triples(chunk)
            new_extractions += 1
            if new_extractions in milestones:
                pct = round(new_extractions / n_todo * 100)
                print(f"  ... {pct}% extracted ({new_extractions}/{n_todo})")
                # save partway through so an interrupt doesn't lose work
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)
        all_triples.extend(
            (t[0], t[1], t[2], src) if with_sources else t
            for t in cache[key]
        )

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"Knowledge graph ready: {len(all_triples)} triples "
          f"({new_extractions} newly extracted, {n_cached} from cache).")
    return all_triples


# ---------- 2. build the graph ----------
def build_graph(triples):
    """Loads triples into a directed NetworkX graph; each edge is labelled with its
    relation and, when available, the source page it came from.

    Accepts either [s, r, o] or [s, r, o, source]. The source is what allows
    graph_facts_for_query to restrict facts to the pages retrieval returned.
    """
    graph = nx.DiGraph()
    for t in triples:
        if len(t) == 4:
            subject, relation, obj, src = t
        else:
            subject, relation, obj = t
            src = None
        graph.add_edge(subject.lower().strip(), obj.lower().strip(),
                       relation=relation, source=src)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


# ---------- 3. find relevant graph facts for a query ----------
def graph_facts_for_query(query, graph, max_facts=8, min_len=5,
                          allowed_sources=None):
    """Returns up to max_facts triples whose entity appears as a whole word in the query.

    allowed_sources: if given (a set of source filenames), only facts extracted
    from those pages are returned. This is the fix for cross-document
    contamination.

    Why it is needed — measured, not assumed. On a 1,000-page corpus the naive
    version degraded +KG below baseline (accuracy 0.657 -> 0.600, faithfulness
    0.714 -> 0.571). Diagnostics showed the cause was NOT OCR noise (only 2% of
    injected facts were garbled) but topical irrelevance: matching any node >=5
    chars against a 50,515-node graph meant common nouns became hubs. A question
    about an instrument developed at Laval University matched the node "patient"
    and received eight true-but-unrelated facts about gingivectomy, chemotherapy
    and knee replacement, pulled from other papers entirely. The model treated
    them as evidence, which is why faithfulness fell hardest.

    Restricting to the retrieved pages makes this genuine late fusion: the KG
    supplies structured facts about the SAME evidence the retriever selected,
    rather than arbitrary facts from across the corpus.
    """
    query_lower = query.lower()
    matched_nodes = []
    for node in graph.nodes:
        if len(node) < min_len:
            continue                                  # skip short, common entity names
        if re.search(r'\b' + re.escape(node) + r'\b', query_lower):
            matched_nodes.append(node)

    facts = []
    for node in matched_nodes:
        for _, obj, data in graph.out_edges(node, data=True):
            if allowed_sources is not None and data.get("source") not in allowed_sources:
                continue
            facts.append(f"{node} {data['relation']} {obj}")
        for subject, _, data in graph.in_edges(node, data=True):
            if allowed_sources is not None and data.get("source") not in allowed_sources:
                continue
            facts.append(f"{subject} {data['relation']} {node}")

    # de-duplicate while preserving order, then cap
    seen, unique = set(), []
    for fact in facts:
        if fact not in seen:
            seen.add(fact)
            unique.append(fact)
    return unique[:max_facts]


# ---------- 4. answer using text chunks + graph facts (late fusion) ----------
def generate_with_graph(query, retrieved, graph_facts, model="gpt4o-mini"):
    """+KG generation: text context plus knowledge-graph facts. Returns GenResult."""
    from llm_client import call
    context = "\n\n".join(f"[{src}] {chunk}" for _, src, chunk in retrieved)
    facts_block = "\n".join(f"- {fact}" for fact in graph_facts) if graph_facts else "(none)"
    return call(
        model,
        system="Answer using only the provided context and knowledge-graph facts. "
               "If the answer isn't there, say you don't know.",
        user=(f"Context passages:\n{context}\n\n"
              f"Knowledge-graph facts:\n{facts_block}\n\n"
              f"Question: {query}"),
    )

def ask_graph(query, text_index, chunks, sources, graph, k=3, model="gpt4o-mini"):
    retrieved = retrieve(query, text_index, chunks, sources, k=k)
    # Restrict KG facts to the pages retrieval actually returned — see
    # graph_facts_for_query for why.
    allowed = {src for _, src, _ in retrieved}
    facts = graph_facts_for_query(query, graph, allowed_sources=allowed)
    answer = generate_with_graph(query, retrieved, facts, model=model).text
    return answer, retrieved, facts


# ---------- interactive demo ----------
if __name__ == "__main__":
    LEVEL = parse_level(default=2)        # explanation verbosity, set with --level N
    text_index, chunks, sources = build_index()
    graph = build_graph(build_triples())

    print(f"\nGraph-enhanced RAG (explain level={LEVEL}). Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q", ""):
            break
        answer, retrieved, facts = ask_graph(question, text_index, chunks, sources, graph)

        # this system used retrieved text + graph facts (no figure)
        evidence = {
            "text": text_evidence(retrieved),
            "graph_facts": facts,
            "reasoning": "Answer drawn from retrieved text, supported by knowledge-graph facts.",
        }
        print("\n" + format_explanation(answer, evidence, LEVEL) + "\n" + "-" * 60)