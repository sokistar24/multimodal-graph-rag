"""Non-graph retrieval arms that test whether a graph gain is graph-specific.

Every arm draws from the chunks the dense index holds and fills the same
candidate budget as +KGret. Two replace the dense ranking: BM25 over the
chunks, and a hybrid that fuses the dense and BM25 rankings by reciprocal
rank. Two keep the dense top-``k`` and fill the remaining slots without
following any graph edge: lexical entity expansion adds the best-ranked chunks
that name an entity matched in the question, and random padding adds random
chunks as a floor for any expansion. Like +KGret, both expansions add at most
one chunk per source and never a source the dense top-``k`` already holds.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .graph import KnowledgeGraph
from .graph_expansion import DEEP_SEARCH_K
from .text import Passage, TextIndex

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
FUSION_DEPTH = 100
RRF_CONSTANT = 60


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, documents: Mapping[str, str], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("documents must be non-empty")
        self.documents = dict(documents)
        self.k1 = k1
        self.b = b
        self.tokens = {key: tokenize(value) for key, value in self.documents.items()}
        self.lengths = {key: len(value) for key, value in self.tokens.items()}
        self.avg_len = sum(self.lengths.values()) / len(self.lengths)
        self.tf = {key: Counter(value) for key, value in self.tokens.items()}
        self.df = Counter()
        for terms in self.tokens.values():
            self.df.update(set(terms))
        self.postings: dict[str, list[str]] = defaultdict(list)
        for key, frequencies in self.tf.items():
            for term in frequencies:
                self.postings[term].append(key)

    def _term_weight(self, term: str, document_id: str) -> float:
        frequency = self.tf[document_id][term]
        inverse_frequency = math.log(
            1 + (len(self.documents) - self.df[term] + 0.5) / (self.df[term] + 0.5)
        )
        denominator = frequency + self.k1 * (
            1 - self.b + self.b * self.lengths[document_id] / max(self.avg_len, 1)
        )
        return inverse_frequency * frequency * (self.k1 + 1) / denominator

    def score(self, query: str, document_id: str) -> float:
        return sum(
            self._term_weight(term, document_id)
            for term in tokenize(query)
            if self.tf[document_id][term]
        )

    def search(self, query: str, k: int = 5) -> list[tuple[float, str]]:
        """Top ``k`` documents by score, ties broken by key.

        Only documents sharing a term with the query are scored. Every other
        document scores zero and fills any remaining places in key order.
        """
        scores: dict[str, float] = defaultdict(float)
        for term in tokenize(query):
            for key in self.postings.get(term, ()):
                scores[key] += self._term_weight(term, key)
        ranked = sorted(
            ((score, key) for key, score in scores.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )[:k]
        if len(ranked) < k:
            unmatched = sorted(set(self.documents) - set(scores))
            ranked.extend((0.0, key) for key in unmatched[: k - len(ranked)])
        return ranked


def fuse_rankings(
    rankings: Sequence[Sequence[Hashable]], constant: int = RRF_CONSTANT
) -> list[tuple[Hashable, float]]:
    """Reciprocal-rank fusion scores, best first, ties broken by item."""
    scores: dict[Hashable, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = 5, constant: int = RRF_CONSTANT
) -> list[Hashable]:
    return [item for item, _ in fuse_rankings(rankings, constant)[:k]]


def chunk_lexicon(index: TextIndex) -> BM25Index:
    """BM25 over the dense index's own chunks, keyed by zero-padded position."""
    width = len(str(index.size))
    return BM25Index(
        {f"{position:0{width}d}": chunk for position, chunk in enumerate(index.chunks)}
    )


def bm25_passages(
    index: TextIndex, lexicon: BM25Index, query: str, k: int
) -> tuple[Passage, ...]:
    return tuple(
        index.passage_at(int(key), score, rank)
        for rank, (score, key) in enumerate(lexicon.search(query, k), 1)
    )


def hybrid_passages(
    index: TextIndex,
    lexicon: BM25Index,
    query: str,
    query_vector: np.ndarray,
    k: int,
    *,
    depth: int = FUSION_DEPTH,
) -> tuple[Passage, ...]:
    """The top ``k`` chunks by reciprocal-rank fusion of dense and BM25 rankings."""
    dense = [position for position, _ in index.search_positions(query_vector, depth)]
    lexical = [int(key) for _, key in lexicon.search(query, depth)]
    fused = fuse_rankings([dense, lexical])[:k]
    return tuple(
        index.passage_at(int(position), score, rank)
        for rank, (position, score) in enumerate(fused, 1)
    )


@dataclass(frozen=True)
class ExpandedRetrieval:
    base: tuple[Passage, ...]
    added: tuple[Passage, ...]

    @property
    def ranked_sources(self) -> list[str]:
        return [p.source for p in self.base] + [p.source for p in self.added]


def lexical_entity_expansion(
    index: TextIndex,
    graph: KnowledgeGraph,
    query: str,
    query_vector: np.ndarray,
    *,
    k: int,
    n_added: int,
    deep_k: int = DEEP_SEARCH_K,
) -> ExpandedRetrieval:
    """Dense top-``k`` plus the best-ranked chunks that name a question entity.

    The entities are the graph nodes that +KGret matches in the question, so
    both arms start from the same names. This arm then looks for those names
    in the chunk text instead of following edges to their provenance, and
    searches the same ``deep_k`` dense candidates that +KGret does.
    """
    ranked = index.search_positions(query_vector, max(k, deep_k))
    base = tuple(
        index.passage_at(position, score, rank)
        for rank, (position, score) in enumerate(ranked[:k], 1)
    )
    patterns = [
        re.compile(rf"\b{re.escape(entity)}\b")
        for entity in graph.match_entities(query)
    ]
    excluded = {passage.source for passage in base}
    added: list[Passage] = []
    for position, score in ranked[k:]:
        if len(added) >= n_added or not patterns:
            break
        source = index.sources[position]
        if source in excluded:
            continue
        text = index.chunks[position].lower()
        if any(pattern.search(text) for pattern in patterns):
            added.append(index.passage_at(position, score, k + len(added) + 1))
            excluded.add(source)
    return ExpandedRetrieval(base, tuple(added))


def random_padding(
    index: TextIndex,
    query_vector: np.ndarray,
    *,
    k: int,
    n_added: int,
    seed: str,
) -> ExpandedRetrieval:
    """Dense top-``k`` plus ``n_added`` random chunks from unretrieved sources."""
    base = tuple(index.search_vector(query_vector, k))
    excluded = {passage.source for passage in base}
    pool = [
        position
        for position, source in enumerate(index.sources)
        if source not in excluded
    ]
    random.Random(seed).shuffle(pool)
    added: list[Passage] = []
    for position in pool:
        if len(added) >= n_added:
            break
        source = index.sources[position]
        if source in excluded:
            continue
        added.append(index.passage_at(position, 0.0, k + len(added) + 1))
        excluded.add(source)
    return ExpandedRetrieval(base, tuple(added))
