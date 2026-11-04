import json
from types import SimpleNamespace

import pytest

from multimodal_graph_rag.clients import (
    GENERATORS,
    JUDGE_ATTEMPTS,
    JUDGE_MAX_TOKENS,
    JUDGES,
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
        self.last_kwargs = kwargs
        self.calls = getattr(self, "calls", []) + [kwargs]
        reply = self.replies.pop(0)
        output_tokens = 5
        if isinstance(reply, tuple):  # (text, output tokens) for a long reply
            reply, output_tokens = reply
        if isinstance(reply, BaseException):
            raise reply
        if reply is None:  # OpenRouter's shape when the upstream provider fails
            return SimpleNamespace(
                choices=None, usage=None, error={"code": 502, "message": "upstream"}
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=output_tokens),
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
    snapshot = load_pricing("configs/pricing_2026-10-20.json")
    assert set(snapshot.models) == set(MODELS)


def test_legacy_pricing_snapshot_still_loads_for_frozen_runs():
    """Runs from the first submission cite the 09-03 snapshot; it must keep loading."""
    snapshot = load_pricing("configs/pricing_2026-09-03.json")
    assert set(snapshot.models) < set(MODELS)


def test_missing_price_is_an_error_not_zero():
    snapshot = PricingSnapshot(
        "2026-01-01", "USD", {"gpt4o-mini": ModelPrice(1.0, 2.0)}
    )
    with pytest.raises(PricingError):
        snapshot.price("deepseek")
    assert ModelPrice(1.0, 2.0).cost(1_000_000, 500_000) == pytest.approx(2.0)


def test_missing_api_key_is_reported_by_name():
    client = ModelClient(_snapshot(), environ={})
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
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
    exhausted = _client([OSError("down")] * 12)
    with pytest.raises(ProviderError, match="12 attempts"):
        exhausted.call("gpt4o-mini", user="q")


def test_a_response_with_no_choices_is_retried_then_reported():
    client = _client([None, "ok"])
    assert client.call("gpt4o-mini", user="q").text == "ok"
    exhausted = _client([None] * 12)
    with pytest.raises(ProviderError, match="no choices.*upstream"):
        exhausted.call("gpt4o-mini", user="q")


def test_judge_reads_the_verdict_from_the_last_line():
    client = _client(["The generated answer names the same person.\n1"])
    value, call = client.judge("deepseek", "grade")
    assert value == 1
    assert call.text.startswith("The generated answer")
    assert client._providers["deepseek"].last_kwargs["max_tokens"] == JUDGE_MAX_TOKENS
    labelled = _client(["It contradicts the reference.\n\n**Grade: 0**"])
    assert labelled.judge("deepseek", "grade")[0] == 0
    assert _client(["0"]).judge("deepseek", "grade")[0] == 0


@pytest.mark.parametrize(
    "reply", ["Sure! 1", "The answer is right, so 1", "Correct.\n1 or 0", ""]
)
def test_judge_rejects_a_reply_without_a_verdict_line(reply):
    with pytest.raises(InvalidModelOutputError):
        _client([reply] * JUDGE_ATTEMPTS).judge("deepseek", "grade")


def test_judge_asks_again_after_a_cut_off_reply_and_keeps_its_cost():
    client = _client(["The answer introduces", "It names the same person.\n1"])
    value, call = client.judge("deepseek", "grade")
    assert value == 1
    assert call.text.startswith("It names")
    # each stub reply costs 10 input and 5 output tokens at $1 / $2 per million
    assert call.cost_usd == pytest.approx(2 * (10 * 1.0 + 5 * 2.0) / 1e6)


def test_judge_switches_reasoning_off_after_a_reply_fills_the_cap():
    client = _client(["The answer", ("", JUDGE_MAX_TOKENS), "It matches.\n1"])
    value, _ = client.judge("deepseek", "grade")
    assert value == 1
    bodies = [call.get("extra_body") for call in client._providers["deepseek"].calls]
    assert bodies == [None, None, {"reasoning": {"enabled": False}}]


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


OPENROUTER_ROSTER = {
    # alias: (OpenRouter model id, vision)
    "gpt4o-mini": ("openai/gpt-4o-mini", True),
    "gemini-flash-lite": ("google/gemini-3.1-flash-lite", True),
    "llama4-maverick": ("meta-llama/llama-4-maverick", True),
    "llama4-scout": ("meta-llama/llama-4-scout", True),
    "gpt4o": ("openai/gpt-4o", True),
    "deepseek": ("deepseek/deepseek-chat", False),
    "claude-haiku": ("anthropic/claude-haiku-4.5", True),
    "text-embedding-3-small": ("openai/text-embedding-3-small", False),
    "qwen3.7-flash": ("qwen/qwen3.7-flash", True),
    "gpt-6.1-sol": ("openai/gpt-6.1-sol", True),
    "claude-opus-5.5": ("anthropic/claude-opus-5.5", True),
    "deepseek-v4-pro": ("deepseek/deepseek-v4-pro-0813", False),
    "grok-4.7": ("x-ai/grok-4.7", True),
    "gemini-3.8-flash": ("google/gemini-3.8-flash", True),
    "qwen3.8-max": ("qwen/qwen3.8-max-0902", True),
    "gpt-6-luna": ("openai/gpt-6-luna", True),
    "mistral-small-4": ("mistralai/mistral-small-2603", True),
    "gemma-4-31b": ("google/gemma-4-31b-it", True),
}


@pytest.mark.parametrize("alias", sorted(OPENROUTER_ROSTER))
def test_openrouter_models_share_one_key_and_endpoint(alias):
    spec = MODELS[alias]
    model_id, vision = OPENROUTER_ROSTER[alias]
    assert spec.model == model_id
    assert spec.sdk == "openai"
    assert spec.api_key_env == "OPENROUTER_API_KEY"
    assert spec.base_url == "https://openrouter.ai/api/v1"
    assert spec.vision is vision


def test_every_registered_model_needs_only_the_openrouter_key():
    assert set(OPENROUTER_ROSTER) == set(MODELS)


def test_pinned_models_send_a_no_fallback_provider_order():
    client = _client(["ok", "ok"])
    client._providers["llama4-scout"] = client._providers["gpt4o-mini"]
    client.call("llama4-scout", user="q")
    stub = client._providers["llama4-scout"]
    assert stub.last_kwargs["extra_body"] == {
        "provider": {"order": ["DeepInfra"], "allow_fallbacks": False}
    }
    client.call("gpt4o-mini", user="q")
    assert "extra_body" not in stub.last_kwargs


def test_current_generators_and_new_judges_are_registered_by_role():
    for generator in (
        "qwen3.7-flash",
        "gemini-3.8-flash",
        "qwen3.8-max",
        "gpt-6-luna",
        "gemma-4-31b",
    ):
        assert generator in GENERATORS
    assert GENERATORS[:4] == (
        "gpt4o-mini",
        "gemini-flash-lite",
        "llama4-maverick",
        "llama4-scout",
    ), "legacy generator order feeds table columns and must not move"
    for judge in (
        "gpt-6.1-sol",
        "claude-opus-5.5",
        "deepseek-v4-pro",
        "grok-4.7",
    ):
        assert judge in JUDGES


def test_openrouter_provider_is_built_against_openrouter():
    client = ModelClient(
        _snapshot(), environ={"OPENROUTER_API_KEY": "or-key"}, sleep=lambda s: None
    )
    provider = client._provider("qwen3.7-flash")
    assert str(provider.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_openrouter_model_without_key_names_the_missing_variable():
    client = ModelClient(
        _snapshot(), environ={"OPENAI_API_KEY": "x"}, sleep=lambda s: None
    )
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        client.api_key("gpt-6.1-sol")
