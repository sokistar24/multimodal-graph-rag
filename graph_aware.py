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

EXTRACT_MODEL = "gpt-4o"
CACHE_FILE = "triples_cache.json"


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

def build_triples(corpus_dir=CORPUS_DIR):
    """Extracts triples for every chunk in the corpus, using the cache where possible,
    and returns the full list of triples."""
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)

    # gather every chunk first so the total is known up front
    chunks_to_process = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*"))):
        text = load_file(path)
        for chunk in chunk_text(text):
            chunks_to_process.append(chunk)

    total = len(chunks_to_process)
    n_cached = sum(1 for c in chunks_to_process if _chunk_key(c) in cache)
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

    for chunk in tqdm(chunks_to_process, desc="Extracting KG triples"):
        key = _chunk_key(chunk)
        if key not in cache:
            cache[key] = extract_triples(chunk)
            new_extractions += 1
            if new_extractions in milestones:
                pct = round(new_extractions / n_todo * 100)
                print(f"  ... {pct}% extracted ({new_extractions}/{n_todo})")
                # save partway through so an interrupt doesn't lose work
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)
        all_triples.extend(cache[key])

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"Knowledge graph ready: {len(all_triples)} triples "
          f"({new_extractions} newly extracted, {n_cached} from cache).")
    return all_triples


# ---------- 2. build the graph ----------
def build_graph(triples):
    """Loads triples into a directed NetworkX graph; each edge is labelled with its relation."""
    graph = nx.DiGraph()
    for subject, relation, obj in triples:
        graph.add_edge(subject.lower().strip(), obj.lower().strip(), relation=relation)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


# ---------- 3. find relevant graph facts for a query ----------
def graph_facts_for_query(query, graph, max_facts=8, min_len=5):
    """Returns up to max_facts triples whose entity appears as a whole word in the query.

    The whole-word match plus a minimum entity length avoids the earlier problem where
    short, common nodes (e.g. "age") matched almost any question and injected irrelevant
    facts."""
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
            facts.append(f"{node} {data['relation']} {obj}")
        for subject, _, data in graph.in_edges(node, data=True):
            facts.append(f"{subject} {data['relation']} {node}")

    # de-duplicate while preserving order, then cap
    seen, unique = set(), []
    for fact in facts:
        if fact not in seen:
            seen.add(fact)
            unique.append(fact)
    return unique[:max_facts]


# ---------- 4. answer using text chunks + graph facts (late fusion) ----------
def generate_with_graph(query, retrieved, graph_facts):
    context = "\n\n".join(f"[{src}] {chunk}" for _, src, chunk in retrieved)
    facts_block = "\n".join(f"- {fact}" for fact in graph_facts) if graph_facts else "(none)"
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[
            {"role": "system",
             "content": "Answer using only the provided context and knowledge-graph facts. "
                        "If the answer isn't there, say you don't know."},
            {"role": "user",
             "content": (f"Context passages:\n{context}\n\n"
                         f"Knowledge-graph facts:\n{facts_block}\n\n"
                         f"Question: {query}")},
        ],
    )
    return resp.choices[0].message.content

def ask_graph(query, text_index, chunks, sources, graph, k=3):
    retrieved = retrieve(query, text_index, chunks, sources, k=k)
    facts = graph_facts_for_query(query, graph)
    answer = generate_with_graph(query, retrieved, facts)
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