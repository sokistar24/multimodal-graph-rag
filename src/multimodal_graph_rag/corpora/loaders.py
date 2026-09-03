"""Document loaders and chunking shared by indexing, graph extraction, and audits."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from docx import Document

from ..errors import ConfigurationError

SUPPORTED_SUFFIXES = {".txt", ".docx", ".csv"}
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class Chunk:
    """A text span and the corpus file it came from."""

    text: str
    source: str


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_docx(path: Path) -> str:
    document = Document(str(path))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def load_csv(path: Path) -> str:
    """Render each row as ``column: value`` pairs so it can be embedded."""
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(" | ".join(f"{key}: {value}" for key, value in row.items()))
    return "\n".join(rows)


def load_file(path: str | Path) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return load_txt(file_path)
    if suffix == ".docx":
        return load_docx(file_path)
    if suffix == ".csv":
        return load_csv(file_path)
    raise ConfigurationError(f"unsupported corpus file type: {file_path}")


def corpus_files(corpus_dir: str | Path) -> list[Path]:
    """Sorted corpus documents; raises if the directory is missing or empty."""
    directory = Path(corpus_dir)
    if not directory.is_dir():
        raise ConfigurationError(f"corpus directory does not exist: {directory}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ConfigurationError(f"corpus directory holds no documents: {directory}")
    return files


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Fixed-width character chunks with a shared margin between neighbours."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ConfigurationError("chunk_size must exceed overlap and both be positive")
    normalised = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(normalised):
        chunks.append(normalised[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def iter_corpus_chunks(
    corpus_dir: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[Chunk]:
    """Every chunk in the corpus, in a deterministic order, with provenance."""
    for path in corpus_files(corpus_dir):
        for text in chunk_text(load_file(path), chunk_size, overlap):
            yield Chunk(text=text, source=path.name)


def corpus_documents(corpus_dir: str | Path) -> dict[str, str]:
    """Full document text keyed by file name, for lexical retrieval controls."""
    return {path.name: load_file(path) for path in corpus_files(corpus_dir)}
