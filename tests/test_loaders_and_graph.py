import json

import pytest

from multimodal_graph_rag.corpora.loaders import (
    chunk_text,
    corpus_files,
    iter_corpus_chunks,
)
from multimodal_graph_rag.errors import (
    CacheError,
    ConfigurationError,
    InvalidModelOutputError,
)
from multimodal_graph_rag.retrieval.graph import (
    KnowledgeGraph,
    Triple,
    TripleStore,
    build_triples,
    load_graph,
    parse_triples,
)
from multimodal_graph_rag.retrieval.graph_expansion import bridge_pages


def test_chunks_overlap_and_keep_provenance(corpus_dir):
    chunks = chunk_text("a" * 1200, chunk_size=500, overlap=50)
    assert [len(c) for c in chunks] == [500, 500, 300]
    assert chunks[1][:50] == chunks[0][-50:]
    sources = {chunk.source for chunk in iter_corpus_chunks(corpus_dir)}
    assert sources == {path.name for path in corpus_files(corpus_dir)}
    assert {"alpha.txt", "beta.txt", "gamma.txt", "delta.txt"} <= sources


def test_missing_or_empty_corpus_is_an_error(tmp_path):
    with pytest.raises(ConfigurationError):
        corpus_files(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigurationError):
        corpus_files(tmp_path / "empty")


def test_parse_triples_rejects_malformed_output():
    assert parse_triples('```json\n[["a","b","c"],["x","y"]]\n```') == [("a", "b", "c")]
    with pytest.raises(InvalidModelOutputError):
        parse_triples("not json")
    with pytest.raises(InvalidModelOutputError):
        parse_triples('{"a": 1}')


def test_corrupt_cache_and_missing_chunks_are_explicit(tmp_path, corpus_dir):
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{", encoding="utf-8")
    with pytest.raises(CacheError):
        TripleStore.load(corrupt)
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(CacheError, match="no cached triples"):
        build_triples(corpus_dir, empty)


def test_graph_facts_are_restricted_to_allowed_sources(corpus_dir, triple_cache):
    graph = load_graph(corpus_dir, triple_cache)
    query = "Which receptor does compound Zeta bind?"
    assert graph.match_entities(query) == ["compound zeta"]
    everything = graph.facts_for_query(query)
    assert "compound zeta binds kappa receptor" in everything
    assert "alpha protocol administered compound zeta" in everything
    only_beta = graph.facts_for_query(query, allowed_sources=["beta.txt"])
    assert only_beta == ["compound zeta binds kappa receptor"]


def test_bridge_pages_reach_second_hop_only_when_asked(corpus_dir, triple_cache):
    graph = load_graph(corpus_dir, triple_cache)
    query = "Who described the receptor that compound Zeta binds?"
    assert bridge_pages(query, graph, hops=1) == {"alpha.txt", "beta.txt"}
    # "kappa receptor" occurs only in beta.txt; its neighbour "compound zeta"
    # also occurs in alpha.txt, which is reachable only with a second hop.
    query_two_hop = "Where is kappa receptor density highest?"
    assert bridge_pages(query_two_hop, graph, hops=1) == {"beta.txt"}
    assert bridge_pages(query_two_hop, graph, hops=2) == {"beta.txt", "alpha.txt"}


def test_statistics_report_overwritten_edges():
    graph = KnowledgeGraph(
        [
            Triple("a", "r1", "b", "x.txt"),
            Triple("A", "r2", "B", "y.txt"),
            Triple("c", "r1", "d", ""),
        ]
    )
    stats = graph.statistics()
    assert stats.triple_count == 3
    assert stats.edge_count == 2
    assert stats.overwritten_edge_count == 1
    assert stats.provenance_coverage == 0.5
    assert stats.relation_count == 2


def test_triple_cache_layout_is_preserved(triple_cache):
    raw = json.loads(triple_cache.read_text(encoding="utf-8"))
    store = TripleStore.load(triple_cache)
    assert store.entries == raw
