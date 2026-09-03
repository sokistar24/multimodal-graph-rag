"""Corpus registry and ingesters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigurationError


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    default_corpus_dir: Path
    default_image_dir: Path | None
    has_images: bool
    provenance_unit: str


CORPORA: dict[str, CorpusSpec] = {
    "publaynet": CorpusSpec(
        "publaynet", Path("publaynet_corpus"), Path("publaynet_images"), True, "page"
    ),
    "spiqa": CorpusSpec(
        "spiqa", Path("spiqa_corpus"), Path("spiqa_images"), True, "paper"
    ),
    "hotpotqa": CorpusSpec(
        "hotpotqa", Path("hotpotqa_corpus"), None, False, "paragraph"
    ),
    "docbank": CorpusSpec(
        "docbank", Path("docbank_corpus"), Path("docbank_images"), True, "page"
    ),
    "doclaynet": CorpusSpec(
        "doclaynet", Path("doclaynet_corpus"), Path("doclaynet_images"), True, "page"
    ),
}


def corpus_spec(name: str) -> CorpusSpec:
    try:
        return CORPORA[name]
    except KeyError as exc:
        raise ConfigurationError(
            f"unsupported corpus: {name!r}; choose from {list(CORPORA)}"
        ) from exc
