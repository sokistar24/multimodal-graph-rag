import json

import numpy as np
import pytest

from multimodal_graph_rag.errors import CacheError
from multimodal_graph_rag.retrieval.text import IndexMetadata, TextIndex

pytest.importorskip("faiss")


def _index():
    vectors = np.eye(3, dtype="float32")
    metadata = IndexMetadata(
        "c", "fp", "text-embedding-3-small", 500, 50, 3, 2, 3, 0, 0.0
    )
    return TextIndex(
        vectors, ["one", "two", "three"], ["a.txt", "a.txt", "b.txt"], metadata
    )


def test_search_returns_ranked_passages_with_provenance():
    index = _index()
    hits = index.search_vector(np.array([0.0, 1.0, 0.0], dtype="float32"), 2)
    assert [(h.source, h.text, h.rank) for h in hits][:1] == [("a.txt", "two", 1)]
    assert index.document_texts() == {"a.txt": "one two", "b.txt": "three"}


def test_cache_round_trip_and_corruption(tmp_path):
    index = _index()
    vectors_path, metadata_path = TextIndex.cache_paths(tmp_path, "corpus", "fp")
    index.save(vectors_path, metadata_path)
    restored = TextIndex.load(vectors_path, metadata_path)
    assert restored.chunks == index.chunks
    assert restored.metadata == index.metadata
    metadata = json.loads(metadata_path.read_text())
    metadata["chunk_count"] = 99
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(CacheError, match="disagrees"):
        TextIndex.load(vectors_path, metadata_path)


def test_build_uses_client_and_reuses_cache(tmp_path, corpus_dir, fake_client):
    index = TextIndex.build(corpus_dir, fake_client, cache_dir=tmp_path / ".cache")
    assert index.size == len(list(corpus_dir.glob("*.txt")))
    assert fake_client.embed_calls == 1
    again = TextIndex.build(corpus_dir, fake_client, cache_dir=tmp_path / ".cache")
    assert fake_client.embed_calls == 1
    assert again.metadata.embedding_tokens == index.metadata.embedding_tokens
    hits = again.retrieve("vitamin D fracture incidence Oslo", fake_client, k=1)
    assert hits[0].source == "gamma.txt"
    with pytest.raises(CacheError, match="different"):
        TextIndex.build(
            corpus_dir,
            fake_client,
            cache_dir=tmp_path / ".cache",
            chunk_size=200,
            overlap=20,
        )
