"""Pure-Python retrieval controls used to test whether gains are graph-specific."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


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

    def score(self, query: str, document_id: str) -> float:
        total = 0.0
        n_documents = len(self.documents)
        document_length = self.lengths[document_id]
        for term in tokenize(query):
            frequency = self.tf[document_id][term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (n_documents - self.df[term] + 0.5) / (self.df[term] + 0.5)
            )
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * document_length / max(self.avg_len, 1)
            )
            total += inverse_frequency * frequency * (self.k1 + 1) / denominator
        return total

    def search(self, query: str, k: int = 5) -> list[tuple[float, str]]:
        scored = [(self.score(query, key), key) for key in self.documents]
        return sorted(scored, key=lambda pair: (-pair[0], pair[1]))[:k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 5, constant: int = 60
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1.0 / (constant + rank)
    return [
        item
        for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
    ]


def lexical_entity_expansion(
    query: str,
    documents: Mapping[str, str],
    base_ranking: Sequence[str],
    k: int = 5,
) -> list[str]:
    """Equal-budget entity/term control that does not use graph edges."""
    chosen = list(dict.fromkeys(base_ranking))[:k]
    query_terms = set(tokenize(query))
    candidates = []
    for source, text in documents.items():
        if source in chosen:
            continue
        overlap = len(query_terms & set(tokenize(text)))
        candidates.append((overlap, source))
    for _, source in sorted(candidates, key=lambda pair: (-pair[0], pair[1])):
        if len(chosen) == k:
            break
        chosen.append(source)
    return chosen


def matched_random_expansion(
    base_ranking: Sequence[str], population: Iterable[str], k: int = 5, seed: int = 0
) -> list[str]:
    chosen = list(dict.fromkeys(base_ranking))[:k]
    remaining = sorted(set(population) - set(chosen))
    random.Random(seed).shuffle(remaining)
    chosen.extend(remaining[: max(0, k - len(chosen))])
    return chosen
