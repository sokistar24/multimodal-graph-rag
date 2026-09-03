"""Knowledge-graph construction from extracted triples and query-time lookup.

Stages
    1. Extraction: a model reads each corpus chunk and returns
       ``[subject, relation, object]`` triples. Results are cached per chunk so
       extraction runs once per corpus.
    2. Graph: the triples become a directed NetworkX graph whose edges carry
       the relation and the source document.
    3. Lookup: graph nodes that appear as whole words in a question are matched
       and their incident triples are rendered as text facts, optionally
       restricted to the documents the retriever already returned.

The generation-stage ``+KG`` system injects these facts into the prompt. The
retrieval-stage ``+KGret`` system (see :mod:`graph_expansion`) uses the same
entity matching to reach documents that dense retrieval missed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from tqdm import tqdm

from ..clients import EXTRACTION_MODEL, ModelClient
from ..corpora.loaders import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
    iter_corpus_chunks,
)
from ..errors import CacheError, InvalidModelOutputError

MIN_ENTITY_LENGTH = 5
MAX_FACTS = 8

EXTRACTION_PROMPT = (
    "Extract the key facts from the TEXT as a list of triples. "
    "Each triple is [subject, relation, object], capturing one relationship. "
    "Use short noun phrases for subject and object. "
    "Respond with ONLY a JSON list, no markdown, e.g. "
    '[["Augustus","was","first Roman emperor"], ["aqueducts","carried","water"]]\n\n'
    "TEXT:\n{chunk}"
)


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    obj: str
    source: str


def chunk_key(text: str) -> str:
    """Stable cache key for a chunk."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def parse_triples(raw: str) -> list[tuple[str, str, str]]:
    """Parse the extractor's reply; malformed output is an explicit error."""
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise InvalidModelOutputError(
            f"triple extractor returned invalid JSON: {cleaned[:120]!r}"
        ) from exc
    if not isinstance(parsed, list):
        raise InvalidModelOutputError("triple extractor did not return a JSON list")
    triples = []
    for item in parsed:
        if not isinstance(item, list) or len(item) != 3:
            continue
        triples.append(tuple(str(part).strip() for part in item))
    return triples


def default_cache_path(corpus_dir: str | Path, cache_root: str | Path) -> Path:
    """``<cache_root>/triples_cache_<corpus directory name>.json``."""
    return Path(cache_root) / f"triples_cache_{Path(corpus_dir).name}.json"


class TripleStore:
    """Per-chunk triple cache with the on-disk layout used by the released runs."""

    def __init__(self, path: str | Path, entries: dict[str, list[list[str]]]) -> None:
        self.path = Path(path)
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path) -> TripleStore:
        cache_path = Path(path)
        if not cache_path.exists():
            return cls(cache_path, {})
        try:
            with cache_path.open(encoding="utf-8") as stream:
                entries = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheError(f"triple cache is unreadable: {cache_path}") from exc
        if not isinstance(entries, dict) or not all(
            isinstance(value, list) for value in entries.values()
        ):
            raise CacheError(f"triple cache has an unexpected layout: {cache_path}")
        return cls(cache_path, entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as stream:
            json.dump(self.entries, stream, indent=2, ensure_ascii=False)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)


def extract_triples(
    chunk: str, client: ModelClient, *, model: str = EXTRACTION_MODEL
) -> list[tuple[str, str, str]]:
    result = client.call(
        model, user=EXTRACTION_PROMPT.format(chunk=chunk), max_tokens=800
    )
    return parse_triples(result.text)


def build_triples(
    corpus_dir: str | Path,
    cache_path: str | Path,
    client: ModelClient | None = None,
    *,
    model: str = EXTRACTION_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    checkpoint_every: int = 200,
) -> list[Triple]:
    """Triples for every chunk in the corpus, extracting only uncached chunks.

    Without a client the corpus must already be fully cached; a missing chunk
    is then an error rather than a silently empty graph.
    """
    store = TripleStore.load(cache_path)
    chunks: list[Chunk] = list(iter_corpus_chunks(corpus_dir, chunk_size, overlap))
    missing = [chunk for chunk in chunks if chunk_key(chunk.text) not in store]
    if missing and client is None:
        raise CacheError(
            f"{len(missing)} of {len(chunks)} chunks in {corpus_dir} have no cached "
            f"triples in {cache_path}; supply a client to extract them"
        )
    if missing:
        for count, chunk in enumerate(
            tqdm(missing, desc="Extracting triples", unit="chunk"), 1
        ):
            store.entries[chunk_key(chunk.text)] = [
                list(triple)
                for triple in extract_triples(chunk.text, client, model=model)
            ]
            if count % checkpoint_every == 0:
                store.save()
        store.save()
    triples = []
    for chunk in chunks:
        for item in store.entries[chunk_key(chunk.text)]:
            if len(item) != 3:
                raise CacheError(
                    f"triple cache {cache_path} holds a malformed triple: {item!r}"
                )
            triples.append(
                Triple(str(item[0]), str(item[1]), str(item[2]), chunk.source)
            )
    return triples


def normalise_entity(value: str) -> str:
    return value.lower().strip()


@dataclass(frozen=True)
class GraphStatistics:
    node_count: int
    edge_count: int
    triple_count: int
    overwritten_edge_count: int
    isolated_node_count: int
    provenance_coverage: float
    relation_count: int
    top_relations: tuple[tuple[str, int], ...]


class KnowledgeGraph:
    """Directed graph whose edges carry a relation label and a source document.

    Parallel edges between the same pair of nodes collapse to the last triple
    seen, matching the graph used for the released runs; the number of
    overwritten edges is reported in :meth:`statistics` so the loss is visible.
    """

    def __init__(self, triples: Iterable[Triple]) -> None:
        self.graph = nx.DiGraph()
        self.triple_count = 0
        self.overwritten_edge_count = 0
        self.relation_counts: Counter[str] = Counter()
        for triple in triples:
            subject = normalise_entity(triple.subject)
            obj = normalise_entity(triple.obj)
            if self.graph.has_edge(subject, obj):
                self.overwritten_edge_count += 1
            self.graph.add_edge(
                subject, obj, relation=triple.relation, source=triple.source
            )
            self.relation_counts[triple.relation] += 1
            self.triple_count += 1
        self._nodes_by_length = sorted(self.graph.nodes, key=len)

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def statistics(self, top: int = 10) -> GraphStatistics:
        with_source = sum(
            1 for _, _, data in self.graph.edges(data=True) if data.get("source")
        )
        return GraphStatistics(
            node_count=self.node_count,
            edge_count=self.edge_count,
            triple_count=self.triple_count,
            overwritten_edge_count=self.overwritten_edge_count,
            isolated_node_count=sum(1 for _ in nx.isolates(self.graph)),
            provenance_coverage=(with_source / self.edge_count)
            if self.edge_count
            else 0.0,
            relation_count=len(self.relation_counts),
            top_relations=tuple(self.relation_counts.most_common(top)),
        )

    def match_entities(self, query: str, min_len: int = MIN_ENTITY_LENGTH) -> list[str]:
        """Graph nodes that appear as whole words in the query."""
        query_lower = query.lower()
        return [
            node
            for node in self.graph.nodes
            if len(node) >= min_len
            and re.search(rf"\b{re.escape(node)}\b", query_lower)
        ]

    def pages_of(self, node: str) -> set[str]:
        pages = set()
        for _, _, data in self.graph.out_edges(node, data=True):
            if data.get("source"):
                pages.add(data["source"])
        for _, _, data in self.graph.in_edges(node, data=True):
            if data.get("source"):
                pages.add(data["source"])
        return pages

    def neighbours(self, node: str) -> set[str]:
        return set(self.graph.successors(node)) | set(self.graph.predecessors(node))

    def facts_for_query(
        self,
        query: str,
        *,
        max_facts: int = MAX_FACTS,
        min_len: int = MIN_ENTITY_LENGTH,
        allowed_sources: Sequence[str] | None = None,
    ) -> list[str]:
        """Rendered triples incident to matched entities, deduplicated and capped.

        ``allowed_sources`` restricts facts to the documents the retriever
        returned. Without it, common nouns act as hubs and pull unrelated facts
        from across the corpus, which measurably reduced faithfulness.
        """
        allowed = None if allowed_sources is None else set(allowed_sources)
        facts: list[str] = []
        for node in self.match_entities(query, min_len):
            for _, obj, data in self.graph.out_edges(node, data=True):
                if allowed is not None and data.get("source") not in allowed:
                    continue
                facts.append(f"{node} {data['relation']} {obj}")
            for subject, _, data in self.graph.in_edges(node, data=True):
                if allowed is not None and data.get("source") not in allowed:
                    continue
                facts.append(f"{subject} {data['relation']} {node}")
        unique = list(dict.fromkeys(facts))
        return unique[:max_facts]


def load_graph(
    corpus_dir: str | Path,
    cache_path: str | Path,
    client: ModelClient | None = None,
    **extraction_options: object,
) -> KnowledgeGraph:
    """Build the knowledge graph for a corpus from its (cached) triples."""
    return KnowledgeGraph(
        build_triples(corpus_dir, cache_path, client, **extraction_options)
    )
