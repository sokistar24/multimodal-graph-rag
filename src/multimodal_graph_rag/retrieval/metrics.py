"""Objective retrieval metrics with single- and multi-source support."""

from __future__ import annotations

import math
from collections.abc import Sequence


def retrieval_metrics(
    ranked_sources: Sequence[str], gold_sources: Sequence[str], k: int
) -> dict[str, float]:
    ranked = list(ranked_sources[:k])
    gold = tuple(dict.fromkeys(gold_sources))
    if not gold:
        raise ValueError("gold_sources must be non-empty")
    found = [source for source in gold if source in ranked]
    recall = len(found) / len(gold)
    complete = float(len(found) == len(gold))
    ranks = [ranked.index(source) + 1 for source in found]
    mrr = 0.0 if not ranks else 1.0 / min(ranks)
    dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return {
        f"recall@{k}": recall,
        "mrr": mrr,
        "completeness": complete,
        f"ndcg@{k}": 0.0 if ideal == 0 else dcg / ideal,
        "candidate_count": float(len(ranked)),
    }
