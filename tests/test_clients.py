import json
from types import SimpleNamespace

import pytest

from multimodal_graph_rag.clients import (
    MODELS,
    ModelClient,
    ModelPrice,
    PricingSnapshot,
    load_pricing,
)
from multimodal_graph_rag.errors import (
    ConfigurationError,
    InvalidModelOutputError,
    PricingError,
    ProviderError,
)


def _snapshot() -> PricingSnapshot:
    return PricingSnapshot(
        "2026-01-01", "USD", {name: ModelPrice(1.0, 2.0) for name in MODELS}
    )


class _StubProvider:
    """Mimics the OpenAI chat and embeddings surface for one scripted reply."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _create(self, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    def _embed(self, **kwargs):
        batch = kwargs["input"]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(text)), 1.0]) for text in batch],
            usage=SimpleNamespace(prompt_tokens=len(batch)),
        )


def _client(replies, environ=None):
    client = ModelClient(
        _snapshot(), environ=environ or {"OPENAI_API_KEY": "x"}, sleep=lambda s: None
    )
    stub = _StubProvider(replies)
    client._providers["gpt4o-mini"] = stub
    client._providers["text-embedding-3-small"] = stub
    client._providers["deepseek"] = stub
    return client


def test_load_pricing_rejects_unknown_model_and_bad_schema(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "currency": "USD",
                "captured_on": "2026-01-01",
                "models": {"nope": {"input_per_million": 1, "output_per_million": 1}},
            }
        )
    )
    with pytest.raises(ConfigurationError, match="unknown model"):
        load_pricing(path)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "currency": "USD",
                "captured_on": "2026-01-01",
                "models": {},
            }
        )
    )
    with pytest.raises(ConfigurationError, match="schema_version"):
        load_pricing(path)


def test_repository_pricing_snapshot_covers_every_registered_model():
    snapshot = load_pricing("configs/pricing_2026-09-03.json")
    assert set(snapshot.models) == set(MODELS)


def test_missing_price_is_an_error_not_zero():
    snapshot = PricingSnapshot(
        "2026-01-01", "USD", {"gpt4o-mini": ModelPrice(1.0, 2.0)}
    )
    with pytest.raises(PricingError):
        snapshot.price("deepseek")
    assert ModelPrice(1.0, 2.0).cost(1_000_000, 500_000) == pytest.approx(2.0)


def test_missing_api_key_is_reported_by_name():
    client = ModelClient(_snapshot(), environ={})
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        client.call("gpt4o-mini", user="hello")


def test_call_records_tokens_and_cost():
    client = _client(["  answer  "])
    result = client.call("gpt4o-mini", user="q")
    assert result.text == "answer"
    assert (result.input_tokens, result.output_tokens) == (10, 5)
    assert result.cost_usd == pytest.approx((10 * 1.0 + 5 * 2.0) / 1_000_000)


def test_transient_errors_are_retried_then_raised():
    client = _client([OSError("blip"), "ok"])
    assert client.call("gpt4o-mini", user="q").text == "ok"
    exhausted = _client([OSError("down")] * 5)
    with pytest.raises(ProviderError, match="5 attempts"):
        exhausted.call("gpt4o-mini", user="q")


def test_judge_rejects_anything_but_a_bare_digit():
    assert _client(["1"]).judge("deepseek", "grade")[0] == 1
    with pytest.raises(InvalidModelOutputError):
        _client(["Sure! 1"]).judge("deepseek", "grade")


def test_images_require_a_vision_model_and_existing_files(tmp_path):
    client = _client(["ok"])
    with pytest.raises(ConfigurationError, match="does not accept image"):
        client.call("deepseek", user="q", images=[tmp_path / "x.png"])
    with pytest.raises(FileNotFoundError):
        client.call("gpt4o-mini", user="q", images=[tmp_path / "missing.png"])


def test_embed_batches_and_prices_tokens():
    client = _client([])
    result = client.embed(["a", "bb", "ccc"], batch_size=2)
    assert result.vectors.shape == (3, 2)
    assert result.input_tokens == 3
    assert result.cost_usd == pytest.approx(3 * 1.0 / 1_000_000)
