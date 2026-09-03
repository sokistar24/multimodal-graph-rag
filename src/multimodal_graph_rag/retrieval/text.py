"""Dense text retrieval over a corpus of local documents with provenance.

The index is built once per corpus and cached on disk, keyed by a content
fingerprint of the corpus files, the chunking parameters, and the embedding
model. A cache that cannot be read or that disagrees with its metadata is an
error; nothing is rebuilt silently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from ..clients import EMBEDDING_MODEL, ModelClient
from ..corpora.loaders import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    corpus_files,
    iter_corpus_chunks,
)
from ..errors import CacheError, ConfigurationError

CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk."""

    source: str
    text: str
    score: float
    rank: int


@dataclass(frozen=True)
class IndexMetadata:
    corpus_dir: str
    fingerprint: str
    embedding_model: str
    chunk_size: int
    overlap: int
    chunk_count: int
    source_count: int
    dimension: int
    embedding_tokens: int
    embedding_cost_usd: float
    format_version: int = CACHE_FORMAT_VERSION


def corpus_fingerprint(corpus_dir: str | Path) -> str:
    """Hash of file names, sizes, and modification times in the corpus."""
    digest = hashlib.sha256()
    for path in corpus_files(corpus_dir):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return digest.hexdigest()[:16]


def normalise_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array


class TextIndex:
    """FAISS inner-product index over normalised chunk embeddings."""

    def __init__(
        self,
        vectors: np.ndarray,
        chunks: Sequence[str],
        sources: Sequence[str],
        metadata: IndexMetadata,
    ) -> None:
        if len(chunks) != len(sources) or vectors.shape[0] != len(chunks):
            raise ConfigurationError("vectors, chunks, and sources must align")
        self.chunks = list(chunks)
        self.sources = list(sources)
        self.metadata = metadata
        self.vectors = normalise_rows(vectors)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def source_names(self) -> set[str]:
        return set(self.sources)

    def search_vector(self, query_vector: np.ndarray, k: int) -> list[Passage]:
        if k < 1:
            raise ConfigurationError("k must be at least 1")
        query = normalise_rows(np.asarray(query_vector).reshape(1, -1))
        scores, indices = self.index.search(query, min(k, self.size))
        passages = []
        for rank, (score, position) in enumerate(
            zip(scores[0], indices[0], strict=True), 1
        ):
            if position < 0:
                continue
            passages.append(
                Passage(
                    source=self.sources[position],
                    text=self.chunks[position],
                    score=float(score),
                    rank=rank,
                )
            )
        return passages

    def retrieve(self, query: str, client: ModelClient, k: int) -> list[Passage]:
        embedded = client.embed([query], model=self.metadata.embedding_model)
        return self.search_vector(embedded.vectors[0], k)

    def document_texts(self) -> dict[str, str]:
        """Reassembled document text per source, for lexical controls."""
        texts: dict[str, list[str]] = {}
        for chunk, source in zip(self.chunks, self.sources, strict=True):
            texts.setdefault(source, []).append(chunk)
        return {source: " ".join(parts) for source, parts in texts.items()}

    @staticmethod
    def cache_paths(
        cache_dir: str | Path, corpus_dir: str | Path, fingerprint: str
    ) -> tuple[Path, Path]:
        stem = f"text_index_{Path(corpus_dir).name}_{fingerprint}"
        base = Path(cache_dir) / "text_index"
        return base / f"{stem}.npz", base / f"{stem}.json"

    @classmethod
    def load(cls, vectors_path: Path, metadata_path: Path) -> TextIndex:
        try:
            with metadata_path.open(encoding="utf-8") as stream:
                metadata = IndexMetadata(**json.load(stream))
            with np.load(vectors_path, allow_pickle=False) as payload:
                vectors = payload["vectors"]
                chunks = payload["chunks"].tolist()
                sources = payload["sources"].tolist()
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise CacheError(f"text index cache is unreadable: {vectors_path}") from exc
        if metadata.format_version != CACHE_FORMAT_VERSION:
            raise CacheError(
                f"text index cache has an unsupported format: {vectors_path}"
            )
        if len(chunks) != metadata.chunk_count or vectors.shape != (
            metadata.chunk_count,
            metadata.dimension,
        ):
            raise CacheError(
                f"text index cache disagrees with its metadata: {vectors_path}"
            )
        return cls(vectors, chunks, sources, metadata)

    def save(self, vectors_path: Path, metadata_path: Path) -> None:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            vectors_path,
            vectors=self.vectors,
            chunks=np.asarray(self.chunks, dtype=object).astype(str),
            sources=np.asarray(self.sources, dtype=object).astype(str),
        )
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(self.metadata.__dict__, stream, indent=2)

    @classmethod
    def build(
        cls,
        corpus_dir: str | Path,
        client: ModelClient,
        *,
        cache_dir: str | Path,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        rebuild: bool = False,
    ) -> TextIndex:
        """Load the cached index for this corpus, or embed and cache it."""
        fingerprint = corpus_fingerprint(corpus_dir)
        vectors_path, metadata_path = cls.cache_paths(
            cache_dir, corpus_dir, fingerprint
        )
        if not rebuild and vectors_path.exists():
            if not metadata_path.exists():
                raise CacheError(f"text index cache has no metadata: {vectors_path}")
            index = cls.load(vectors_path, metadata_path)
            if index.metadata.embedding_model != embedding_model or (
                index.metadata.chunk_size,
                index.metadata.overlap,
            ) != (chunk_size, overlap):
                raise CacheError(
                    f"text index cache {vectors_path} was built with different "
                    "embedding or chunking settings; pass rebuild=True"
                )
            return index

        chunks = list(iter_corpus_chunks(corpus_dir, chunk_size, overlap))
        texts = [chunk.text for chunk in chunks]
        embedded = client.embed(texts, model=embedding_model)
        metadata = IndexMetadata(
            corpus_dir=str(corpus_dir),
            fingerprint=fingerprint,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
            chunk_count=len(chunks),
            source_count=len({chunk.source for chunk in chunks}),
            dimension=int(embedded.vectors.shape[1]),
            embedding_tokens=embedded.input_tokens,
            embedding_cost_usd=embedded.cost_usd,
        )
        index = cls(
            embedded.vectors, texts, [chunk.source for chunk in chunks], metadata
        )
        index.save(vectors_path, metadata_path)
        return index
