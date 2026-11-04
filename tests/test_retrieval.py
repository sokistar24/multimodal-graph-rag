import pytest

from multimodal_graph_rag.retrieval.baselines import (
    BM25Index,
    bm25_passages,
    chunk_lexicon,
    hybrid_passages,
    lexical_entity_expansion,
    random_padding,
    reciprocal_rank_fusion,
)
from multimodal_graph_rag.retrieval.graph import load_graph
from multimodal_graph_rag.retrieval.metrics import retrieval_metrics
from multimodal_graph_rag.retrieval.text import TextIndex

ZETA_QUESTION = "Which receptor does compound Zeta bind, and who first described it?"


@pytest.fixture
def text_index(tmp_path, corpus_dir, fake_client):
    return TextIndex.build(corpus_dir, fake_client, cache_dir=tmp_path / ".cache")


def _vector(fake_client, text):
    return fake_client.embed([text]).vectors[0]


def test_multi_source_metrics_distinguish_partial_and_complete():
    metrics = retrieval_metrics(["a", "x", "b"], ["a", "b"], 2)
    assert metrics["recall@2"] == 0.5
    assert metrics["completeness"] == 0.0
    assert metrics["mrr"] == 1.0


def test_bm25_and_rrf_are_deterministic():
    index = BM25Index({"a": "graph evidence retrieval", "b": "visual document image"})
    assert index.search("graph retrieval", 1)[0][1] == "a"
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=2) == ["a", "b"]


def test_bm25_search_matches_scoring_every_document():
    documents = {
        "a": "graph evidence retrieval graph",
        "b": "visual document image",
        "c": "graph image",
        "d": "unrelated words here",
    }
    index = BM25Index(documents)
    query = "graph image graph"
    every = sorted(
        ((index.score(query, key), key) for key in documents),
        key=lambda pair: (-pair[0], pair[1]),
    )
    assert index.search(query, 4) == every
    assert index.search(query, 2) == every[:2]


def test_lexical_arms_rank_the_dense_index_chunks(text_index, fake_client):
    lexicon = chunk_lexicon(text_index)
    bm25 = bm25_passages(text_index, lexicon, ZETA_QUESTION, 5)
    assert [p.rank for p in bm25] == [1, 2, 3, 4, 5]
    assert all(p.text in text_index.chunks for p in bm25)
    assert bm25[0].source in {"alpha.txt", "beta.txt"}
    hybrid = hybrid_passages(
        text_index, lexicon, ZETA_QUESTION, _vector(fake_client, ZETA_QUESTION), 5
    )
    assert len(hybrid) == 5
    assert len({p.text for p in hybrid}) == 5
    assert hybrid[0].score >= hybrid[-1].score


def test_lexical_expansion_adds_only_unretrieved_chunks_naming_an_entity(
    text_index, fake_client, corpus_dir, triple_cache
):
    graph = load_graph(corpus_dir, triple_cache)
    retrieval = lexical_entity_expansion(
        text_index,
        graph,
        ZETA_QUESTION,
        _vector(fake_client, ZETA_QUESTION),
        k=1,
        n_added=2,
    )
    base_sources = {p.source for p in retrieval.base}
    assert len(retrieval.base) == 1
    assert 1 <= len(retrieval.added) <= 2
    for rank, passage in enumerate(retrieval.added, 2):
        assert "compound zeta" in passage.text.lower()
        assert passage.source not in base_sources
        assert passage.rank == rank
    assert len({p.source for p in retrieval.added}) == len(retrieval.added)


def test_lexical_expansion_without_a_matched_entity_adds_nothing(
    text_index, fake_client, corpus_dir, triple_cache
):
    graph = load_graph(corpus_dir, triple_cache)
    question = "How long was the winter timetable?"
    retrieval = lexical_entity_expansion(
        text_index, graph, question, _vector(fake_client, question), k=3, n_added=2
    )
    assert len(retrieval.base) == 3
    assert retrieval.added == ()


def test_random_padding_fills_the_budget_reproducibly(text_index, fake_client):
    vector = _vector(fake_client, ZETA_QUESTION)
    first = random_padding(text_index, vector, k=3, n_added=2, seed="7:q1")
    again = random_padding(text_index, vector, k=3, n_added=2, seed="7:q1")
    assert first == again
    assert len(first.ranked_sources) == 5
    assert len(set(first.ranked_sources)) == 5
    assert [p.rank for p in first.added] == [4, 5]
