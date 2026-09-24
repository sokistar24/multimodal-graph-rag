import json

import pytest

from multimodal_graph_rag.errors import CacheError
from multimodal_graph_rag.evaluation.graph_grounding import (
    audit_triple_grounding,
    canonicalisation_collisions,
)
from multimodal_graph_rag.retrieval.graph import chunk_key


@pytest.fixture
def grounded_corpus(tmp_path):
    """One chunk with a grounded triple, a half-grounded one, and a fabricated one."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "Aspirin reduces inflammation in rats during the trial."
    (corpus / "alpha.txt").write_text(text, encoding="utf-8")
    cache = tmp_path / "triples.json"
    cache.write_text(
        json.dumps(
            {
                chunk_key(text): [
                    ["Aspirin", "reduces", "inflammation"],
                    ["Aspirin", "was tested in", "beagles"],
                    ["Ibuprofen", "reduces", "fever"],
                ]
            }
        ),
        encoding="utf-8",
    )
    return corpus, cache


def test_grounding_separates_present_from_fabricated_endpoints(grounded_corpus):
    corpus, cache = grounded_corpus
    report = audit_triple_grounding(corpus, cache, corpus="test")

    assert report.triples == 3
    assert report.both_present == 1  # only the first triple is fully grounded
    assert report.subject_present == 2  # "aspirin" twice, "ibuprofen" never
    assert report.neither_present == 1  # the wholly fabricated triple
    assert report.grounding_rate == pytest.approx(1 / 3)


def test_relation_presence_is_tracked_separately(grounded_corpus):
    corpus, cache = grounded_corpus
    report = audit_triple_grounding(corpus, cache, corpus="test")
    # "reduces" is in the text; "was tested in" is not.
    assert report.relation_present == 2


def test_a_cache_from_another_corpus_is_an_error(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.txt").write_text("Some unrelated sentence.", encoding="utf-8")
    cache = tmp_path / "triples.json"
    cache.write_text(json.dumps({"deadbeef": [["a", "b", "c"]]}), encoding="utf-8")
    with pytest.raises(CacheError, match="different corpus"):
        audit_triple_grounding(corpus, cache)


def test_empty_cache_is_an_error_not_a_zero_rate(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.txt").write_text("Anything.", encoding="utf-8")
    cache = tmp_path / "triples.json"
    cache.write_text("{}", encoding="utf-8")
    with pytest.raises(CacheError, match="empty"):
        audit_triple_grounding(corpus, cache)


def test_collisions_expose_surface_forms_merged_into_one_node():
    entries = {
        "k": [
            ["Aspirin", "reduces", "Inflammation"],
            ["aspirin", "reduces", "inflammation"],
            ["Ibuprofen", "reduces", "fever"],
        ]
    }
    collisions = canonicalisation_collisions(entries)
    assert collisions["aspirin"] == ["Aspirin", "aspirin"]
    assert "ibuprofen" not in collisions  # only one surface form
