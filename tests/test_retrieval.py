from multimodal_graph_rag.retrieval.baselines import (
    BM25Index,
    matched_random_expansion,
    reciprocal_rank_fusion,
)
from multimodal_graph_rag.retrieval.metrics import retrieval_metrics


def test_multi_source_metrics_distinguish_partial_and_complete():
    metrics = retrieval_metrics(["a", "x", "b"], ["a", "b"], 2)
    assert metrics["recall@2"] == 0.5
    assert metrics["completeness"] == 0.0
    assert metrics["mrr"] == 1.0


def test_bm25_and_rrf_are_deterministic():
    index = BM25Index({"a": "graph evidence retrieval", "b": "visual document image"})
    assert index.search("graph retrieval", 1)[0][1] == "a"
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=2) == ["a", "b"]


def test_random_expansion_respects_total_budget():
    result = matched_random_expansion(
        ["a", "b", "c"], ["a", "b", "c", "d", "e"], k=5, seed=9
    )
    assert result[:3] == ["a", "b", "c"]
    assert len(result) == 5
