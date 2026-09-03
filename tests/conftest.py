"""Shared fixtures: a deterministic offline model client and a tiny corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from multimodal_graph_rag.clients import (
    MODELS,
    EmbeddingResult,
    GenResult,
    ModelClient,
    ModelPrice,
    PricingSnapshot,
)
from multimodal_graph_rag.corpora.loaders import iter_corpus_chunks
from multimodal_graph_rag.errors import ConfigurationError, ProviderError
from multimodal_graph_rag.retrieval.graph import chunk_key

EMBEDDING_DIMENSION = 64

DOCUMENTS = {
    "alpha.txt": (
        "The Alpha protocol was introduced by Dr Vega in 2011 at the Lisbon "
        "workshop. Alpha protocol participants received 40 mg of compound Zeta "
        "daily for six weeks. The Lisbon cohort reported fatigue as the most "
        "common side effect."
    ),
    "beta.txt": (
        "Compound Zeta binds the kappa receptor. The kappa receptor was first "
        "described in the Beta study of 1998 by Dr Ortega. Kappa receptor "
        "density is highest in the hippocampus."
    ),
    "gamma.txt": (
        "The Gamma trial compared placebo with vitamin D in 300 adults from "
        "Oslo. Vitamin D reduced fracture incidence by 12 percent over two "
        "years in the Oslo group."
    ),
    "delta.txt": (
        "Ocean acidification affects coral reefs near Fiji. Reef cover near "
        "Fiji declined 8 percent between 2005 and 2015 according to survey "
        "transects."
    ),
    # Distractors: no question asks about these, so they give the shuffled
    # control a pool of unrelated passages of comparable length to draw from.
    "epsilon.txt": (
        "Seismic retrofitting of masonry towers in Bologna used carbon-fibre "
        "wraps. Displacement at the tower crown fell by 19 percent under the "
        "design earthquake."
    ),
    "zeta.txt": (
        "The Helsinki bus network was rescheduled with a mixed-integer solver. "
        "Average passenger waiting time dropped from 7.4 to 5.9 minutes across "
        "the winter timetable."
    ),
    "eta.txt": (
        "Sourdough fermentation at 24 degrees Celsius raised total titratable "
        "acidity to 9.2 millilitres. Loaf volume was largest at a 65 percent "
        "hydration ratio."
    ),
    "theta.txt": (
        "Lidar surveys of the Atacama alluvial fans mapped 41 abandoned "
        "channels. Channel abandonment clustered in the late Pleistocene "
        "according to cosmogenic dating."
    ),
}

TRIPLES = {
    "alpha.txt": [
        ["alpha protocol", "introduced by", "dr vega"],
        ["alpha protocol", "administered", "compound zeta"],
    ],
    "beta.txt": [
        ["compound zeta", "binds", "kappa receptor"],
        ["kappa receptor", "described in", "beta study"],
    ],
    "gamma.txt": [["gamma trial", "compared", "vitamin d"]],
    "delta.txt": [["ocean acidification", "affects", "coral reefs"]],
}

QUESTIONS = [
    {
        "q": "Who introduced the Alpha protocol, and where?",
        "source": "alpha.txt",
        "answer": "Dr Vega, at the Lisbon workshop",
        "type": "text",
    },
    {
        "q": "Which receptor does compound Zeta bind, and who first described it?",
        "source": "alpha.txt",
        "source2": "beta.txt",
        "answer": "the kappa receptor, described by Dr Ortega",
        "type": "multihop_cross",
        "entity": "compound zeta",
    },
    {
        "q": "By how much did vitamin D reduce fracture incidence in the Oslo adults?",
        "source": "gamma.txt",
        "answer": "12 percent",
        "type": "text",
    },
]


def bag_of_words_vector(text: str) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIMENSION, dtype="float32")
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vector[digest % EMBEDDING_DIMENSION] += 1.0
    if not vector.any():
        vector[0] = 1.0
    return vector


class FakeModelClient(ModelClient):
    """Deterministic client that never touches the network."""

    def __init__(
        self,
        *,
        fail_after_calls: int | None = None,
        answer: str = "Dr Vega at the Lisbon workshop",
    ) -> None:
        pricing = PricingSnapshot(
            captured_on="2026-01-01",
            currency="USD",
            models={name: ModelPrice(1.0, 2.0) for name in MODELS},
        )
        super().__init__(pricing, environ={}, sleep=lambda seconds: None)
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []
        self.embed_calls = 0
        self.fail_after_calls = fail_after_calls
        self.answer = answer

    def call(
        self, name, *, user, system="", images=(), max_tokens=800, temperature=0.0
    ) -> GenResult:
        spec = self.spec(name)
        if images and not spec.vision:
            raise ConfigurationError(f"model {name} does not accept image input")
        self.calls.append((name, user, tuple(str(i) for i in images)))
        if (
            self.fail_after_calls is not None
            and len(self.calls) > self.fail_after_calls
        ):
            raise ProviderError(f"{name} failed after 5 attempts: simulated outage")
        price = self.pricing.price(name)
        input_tokens = max(1, len(user) // 4)
        return GenResult(
            text=self.answer,
            model=name,
            input_tokens=input_tokens,
            output_tokens=8,
            latency_ms=1.5,
            cost_usd=price.cost(input_tokens, 8),
        )

    def judge(self, name, prompt, images=()) -> tuple[int, GenResult]:
        result = self.call(name, user=prompt, images=images, max_tokens=8)
        return 1, result

    def embed(
        self, texts: Sequence[str], *, model="text-embedding-3-small", batch_size=1000
    ) -> EmbeddingResult:
        self.embed_calls += 1
        vectors = np.stack([bag_of_words_vector(text) for text in texts])
        tokens = sum(len(text) // 4 for text in texts)
        return EmbeddingResult(
            vectors=vectors,
            input_tokens=tokens,
            cost_usd=self.pricing.price(model).cost(tokens, 0),
        )


@pytest.fixture
def fake_client() -> FakeModelClient:
    return FakeModelClient()


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir()
    for name, text in DOCUMENTS.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


@pytest.fixture
def triple_cache(tmp_path: Path, corpus_dir: Path) -> Path:
    """A fully populated triple cache in the on-disk layout used by the released runs."""
    cache: dict[str, list[list[str]]] = {}
    for chunk in iter_corpus_chunks(corpus_dir):
        cache.setdefault(chunk_key(chunk.text), [])
    for chunk in iter_corpus_chunks(corpus_dir):
        cache[chunk_key(chunk.text)] = TRIPLES.get(chunk.source, [])
    path = tmp_path / "graphs" / "triples_cache_corpus.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(cache), encoding="utf-8")
    return path


@pytest.fixture
def question_file(tmp_path: Path) -> Path:
    path = tmp_path / "questions_test_set.json"
    path.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    return path
