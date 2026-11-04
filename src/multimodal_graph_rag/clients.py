"""Provider registry, frozen pricing, and the single model-call seam.

Every generator, judge, question author, captioner, triple extractor, and
embedding request in the project goes through :class:`ModelClient`. The client
records tokens, latency, and cost for each call from a pricing snapshot that is
loaded explicitly; a model without a frozen price cannot be charged, and a
provider failure that survives the retry budget is raised, never converted into
an empty answer.

Every model is served through OpenRouter with one key.

Generators, all vision-capable. The revision configs run the last five; the
first four are kept for the submitted paper's runs and the pilots only.
    gpt4o-mini          OpenAI      closed  legacy only (shut down 2027-01-14)
    llama4-maverick     Meta        open    legacy only, pinned to Parasail (FP8)
    gemini-flash-lite   Google      closed  legacy only
    qwen3.7-flash       Alibaba     closed  pilots only
    llama4-scout        Meta        open    legacy, pinned to DeepInfra
    gpt-6-luna          OpenAI      closed  current, low cost; OpenAI's named
                                            replacement for gpt4o-mini
    gemma-4-31b         Google      open    current, low cost, pinned to Crusoe (bf16)
    gemini-3.8-flash    Google      closed  current, strong tier
    qwen3.8-max         Alibaba     closed  current, strong tier
The Gemini, Qwen 3.8, and Luna models think before answering; thinking tokens
are billed as output tokens and are left at the provider default.

Diagnostic generator (never part of the reported comparison)
    gpt4o               OpenAI

Judges
    gpt-6.1-sol         OpenAI      primary judge on the figure sets; shares a
                                    family with gpt4o-mini (a stated limitation)
    deepseek-v4-pro     DeepSeek    text judge
    grok-4.7            xAI         second vision judge
    claude-opus-5.5     Anthropic   adjudicator on disagreements
    deepseek            DeepSeek    legacy text judge and text question author
    claude-haiku        Anthropic   legacy vision judge and figure question author

Embeddings
    text-embedding-3-small  OpenAI
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import anthropic
import numpy as np
import openai
from PIL import Image

from .errors import (
    ConfigurationError,
    InvalidModelOutputError,
    PricingError,
    ProviderError,
)


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the provider registry."""

    name: str
    sdk: str
    model: str
    api_key_env: str
    vendor: str
    access: str
    vision: bool
    base_url: str | None = None
    # OpenRouter upstream providers to use, in order, with fallback disabled, so
    # a model is always served by the same backend and quantisation.
    providers: tuple[str, ...] = ()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS: dict[str, ModelSpec] = {
    # Every model is served through OpenRouter (single OPENROUTER_API_KEY,
    # adopted 2026-10-20). The aliases are unchanged from the legacy direct
    # providers so run files, caches, and figures stay comparable.
    "gpt4o-mini": ModelSpec(
        name="gpt4o-mini",
        sdk="openai",
        model="openai/gpt-4o-mini",
        api_key_env="OPENROUTER_API_KEY",
        vendor="OpenAI/OpenRouter",
        access="closed",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    "gemini-flash-lite": ModelSpec(
        name="gemini-flash-lite",
        sdk="openai",
        model="google/gemini-3.1-flash-lite",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Google/OpenRouter",
        access="closed",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    "llama4-maverick": ModelSpec(
        name="llama4-maverick",
        sdk="openai",
        model="meta-llama/llama-4-maverick",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Meta/OpenRouter",
        access="open",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
        # DeepInfra (the legacy backend) no longer serves Maverick on
        # OpenRouter; Parasail serves the same FP8 build.
        providers=("Parasail",),
    ),
    "llama4-scout": ModelSpec(
        name="llama4-scout",
        sdk="openai",
        model="meta-llama/llama-4-scout",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Meta/OpenRouter",
        access="open",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
        providers=("DeepInfra",),
    ),
    "gpt4o": ModelSpec(
        name="gpt4o",
        sdk="openai",
        model="openai/gpt-4o",
        api_key_env="OPENROUTER_API_KEY",
        vendor="OpenAI/OpenRouter",
        access="diagnostic",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    "deepseek": ModelSpec(
        name="deepseek",
        sdk="openai",
        model="deepseek/deepseek-chat",
        api_key_env="OPENROUTER_API_KEY",
        vendor="DeepSeek/OpenRouter",
        access="judge",
        vision=False,
        base_url=OPENROUTER_BASE_URL,
    ),
    "claude-haiku": ModelSpec(
        name="claude-haiku",
        sdk="openai",
        model="anthropic/claude-haiku-4.5",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Anthropic/OpenRouter",
        access="judge",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    "text-embedding-3-small": ModelSpec(
        name="text-embedding-3-small",
        sdk="openai",
        model="openai/text-embedding-3-small",
        api_key_env="OPENROUTER_API_KEY",
        vendor="OpenAI/OpenRouter",
        access="embedding",
        vision=False,
        base_url=OPENROUTER_BASE_URL,
    ),
    # --- OpenRouter roster, chosen 2026-09-10 (docs/REVISION_PLAN.md section 15);
    # judges changed on 2026-10-20: GPT-6.1 Sol replaces Claude Sonnet 5 as
    # primary judge, Opus 5.5 replaces Opus 5, Grok 4.7 replaces Grok 4.3.
    # One key covers every family below; OpenRouter is OpenAI-compatible.
    "qwen3.7-flash": ModelSpec(
        name="qwen3.7-flash",
        sdk="openai",
        model="qwen/qwen3.7-flash",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Alibaba/OpenRouter",
        # API-only: OpenRouter lists no released weights for it, unlike the
        # open Qwen 3.8 checkpoints.
        access="closed",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    # The revision's open-weight model from 2026-10-21, in place of
    # mistral-small-4: Mistral is that model's only provider and OpenRouter's
    # shared capacity for it stayed rate-limited for over six minutes under
    # load. Gemma 4 31B has fifteen providers. It is restricted to the two that
    # serve the unquantised (bf16) weights at the same price, Crusoe then
    # Novita, with no fallback beyond them: pinned to Crusoe alone, a run
    # stopped after six minutes of upstream 429s.
    "gemma-4-31b": ModelSpec(
        name="gemma-4-31b",
        sdk="openai",
        model="google/gemma-4-31b-it",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Google/OpenRouter",
        access="open",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
        providers=("Crusoe", "Novita"),
    ),
    # Registered 2026-10-21 but never run (see gemma-4-31b).
    "mistral-small-4": ModelSpec(
        name="mistral-small-4",
        sdk="openai",
        model="mistralai/mistral-small-2603",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Mistral/OpenRouter",
        access="open",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
        providers=("Mistral",),
    ),
    # Added 2026-10-21. OpenAI shuts gpt4o-mini down on 2027-01-14 and names
    # GPT-6 Luna as its replacement. Same family as the gpt-6.1-sol judge, a
    # limitation stated in the paper.
    "gpt-6-luna": ModelSpec(
        name="gpt-6-luna",
        sdk="openai",
        model="openai/gpt-6-luna",
        api_key_env="OPENROUTER_API_KEY",
        vendor="OpenAI/OpenRouter",
        access="closed",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    # Strong-tier generators, added 2026-10-20 so the roster is not limited to
    # 2024-25 models. Neither family judges, so the judge rules are unchanged.
    "gemini-3.8-flash": ModelSpec(
        name="gemini-3.8-flash",
        sdk="openai",
        model="google/gemini-3.8-flash",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Google/OpenRouter",
        access="closed",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    "qwen3.8-max": ModelSpec(
        name="qwen3.8-max",
        sdk="openai",
        model="qwen/qwen3.8-max-0902",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Alibaba/OpenRouter",
        access="closed",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    "gpt-6.1-sol": ModelSpec(
        name="gpt-6.1-sol",
        sdk="openai",
        model="openai/gpt-6.1-sol",
        api_key_env="OPENROUTER_API_KEY",
        vendor="OpenAI/OpenRouter",
        access="judge",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    "claude-opus-5.5": ModelSpec(
        name="claude-opus-5.5",
        sdk="openai",
        model="anthropic/claude-opus-5.5",
        api_key_env="OPENROUTER_API_KEY",
        vendor="Anthropic/OpenRouter",
        access="judge",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
    "deepseek-v4-pro": ModelSpec(
        name="deepseek-v4-pro",
        sdk="openai",
        model="deepseek/deepseek-v4-pro-0813",
        api_key_env="OPENROUTER_API_KEY",
        vendor="DeepSeek/OpenRouter",
        access="judge",
        vision=False,
        base_url=OPENROUTER_BASE_URL,
    ),
    "grok-4.7": ModelSpec(
        name="grok-4.7",
        sdk="openai",
        model="x-ai/grok-4.7",
        api_key_env="OPENROUTER_API_KEY",
        vendor="xAI/OpenRouter",
        access="judge",
        vision=True,
        base_url=OPENROUTER_BASE_URL,
    ),
}

GENERATORS: tuple[str, ...] = (
    "gpt4o-mini",
    "gemini-flash-lite",
    "llama4-maverick",
    "llama4-scout",
    "qwen3.7-flash",
    "gemini-3.8-flash",
    "qwen3.8-max",
    "gpt-6-luna",
    "gemma-4-31b",
)
DIAGNOSTICS: tuple[str, ...] = ("gpt4o",)
JUDGES: tuple[str, ...] = (
    "deepseek",
    "claude-haiku",
    "gpt-6.1-sol",
    "claude-opus-5.5",
    "deepseek-v4-pro",
    "grok-4.7",
)
EMBEDDING_MODEL = "text-embedding-3-small"
CAPTION_MODEL = "gpt4o"
EXTRACTION_MODEL = "gpt4o-mini"
# Room for a one-sentence justification and, on reasoning judges, the hidden
# reasoning that OpenRouter counts against the same cap. DeepSeek V4 Pro used
# up to 715 tokens in the 2026-10-20 pilot and once exhausted 1,024, returning
# an empty reply; only tokens actually used are billed. On the full bridge run
# (2026-10-21) it returned three empty replies in a row for one Gemma 4 31B
# answer under a 4,096 cap, so the cap is 8,192.
JUDGE_MAX_TOKENS = 8192
# DeepSeek V4 Pro once returned a three-word reply with no verdict on the full
# HotpotQA run (2026-10-21); such replies are asked again before the run stops.
# On the SPIQA cross-paper run (2026-10-22) it spent the whole cap reasoning, at
# temperature 0 and on every attempt, over a question whose reference answer is
# a placeholder; OpenRouter does not hold DeepSeek to a reasoning budget, so a
# reply that fills the cap is asked again with reasoning switched off.
JUDGE_ATTEMPTS = 3
# Generator replies share the cap with hidden reasoning in the same way. Under
# the earlier cap of 800, Qwen 3.8 Max (and GPT-6 Luna without context) often
# spent all of it reasoning and returned an empty answer on the HotpotQA and
# SPIQA cross-paper runs of 2026-10-21/22; only tokens actually used are billed.
GENERATOR_MAX_TOKENS = 8192
RETRY_MAX_SECONDS = 60.0
VERDICT_LINE = re.compile(r"^\W*(?:(?:grade|verdict)\W*)?([01])\W*$", re.IGNORECASE)


class EmptyResponseError(Exception):
    """A completion with no choices, which OpenRouter returns when the upstream
    provider fails mid-request; the error it carries is kept in the message."""


RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    EmptyResponseError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    OSError,
)
PROVIDER_ERRORS: tuple[type[BaseException], ...] = (
    openai.OpenAIError,
    anthropic.AnthropicError,
    OSError,
)


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class PricingSnapshot:
    """Prices frozen on a given date; the date travels into every run record."""

    captured_on: str
    currency: str
    models: Mapping[str, ModelPrice]

    def price(self, name: str) -> ModelPrice:
        try:
            return self.models[name]
        except KeyError as exc:
            raise PricingError(
                f"no frozen price for model {name!r} in the snapshot captured "
                f"on {self.captured_on}"
            ) from exc


def load_pricing(path: str | Path) -> PricingSnapshot:
    """Read and validate a pricing snapshot file."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if raw.get("schema_version") != 1:
        raise ConfigurationError(f"pricing snapshot {path} must use schema_version 1")
    if raw.get("currency") != "USD":
        raise ConfigurationError(f"pricing snapshot {path} must be priced in USD")
    captured_on = str(raw.get("captured_on", "")).strip()
    if not captured_on:
        raise ConfigurationError(f"pricing snapshot {path} has no captured_on date")
    models: dict[str, ModelPrice] = {}
    for name, entry in dict(raw.get("models", {})).items():
        if name not in MODELS:
            raise ConfigurationError(f"pricing snapshot names unknown model {name!r}")
        try:
            price = ModelPrice(
                float(entry["input_per_million"]), float(entry["output_per_million"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"pricing entry for {name!r} needs numeric input_per_million and "
                "output_per_million"
            ) from exc
        if price.input_per_million < 0 or price.output_per_million < 0:
            raise ConfigurationError(f"pricing entry for {name!r} is negative")
        models[name] = price
    if not models:
        raise ConfigurationError(f"pricing snapshot {path} lists no models")
    return PricingSnapshot(captured_on=captured_on, currency="USD", models=models)


@dataclass(frozen=True)
class GenResult:
    """One completed model call with the metadata the efficiency tables need."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    input_tokens: int
    cost_usd: float


def encode_image(path: str | Path, max_px: int = 1200) -> str:
    """Base64-encode an image, downscaling so that no side exceeds ``max_px``."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_px:
        image.thumbnail((max_px, max_px))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ModelClient:
    """Cached provider connections plus the one call path used everywhere."""

    def __init__(
        self,
        pricing: PricingSnapshot,
        *,
        # Twelve attempts, each wait capped at RETRY_MAX_SECONDS, ride out about
        # six minutes of upstream 429s on a model pinned to a single provider;
        # Mistral Small 4 returned them for over two minutes on 2026-10-21.
        max_retries: int = 12,
        retry_base_seconds: float = 2.0,
        environ: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 1:
            raise ConfigurationError("max_retries must be at least 1")
        self.pricing = pricing
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self._environ = dict(os.environ if environ is None else environ)
        self._sleep = sleep
        self._providers: dict[str, object] = {}

    @staticmethod
    def spec(name: str) -> ModelSpec:
        try:
            return MODELS[name]
        except KeyError as exc:
            raise ConfigurationError(f"unknown model alias: {name!r}") from exc

    def api_key(self, name: str) -> str:
        spec = self.spec(name)
        key = self._environ.get(spec.api_key_env, "").strip()
        if not key:
            raise ConfigurationError(
                f"{spec.api_key_env} is not set; it is required to call {name}"
            )
        return key

    def _provider(self, name: str) -> object:
        if name not in self._providers:
            spec = self.spec(name)
            key = self.api_key(name)
            if spec.sdk == "anthropic":
                self._providers[name] = anthropic.Anthropic(api_key=key)
            elif spec.sdk == "openai":
                self._providers[name] = openai.OpenAI(
                    api_key=key, base_url=spec.base_url
                )
            else:
                raise ConfigurationError(f"unsupported sdk {spec.sdk!r} for {name}")
        return self._providers[name]

    def _with_retries(self, name: str, request: Callable[[], GenResult]) -> GenResult:
        last_error: BaseException | None = None
        for attempt in range(self.max_retries):
            try:
                return request()
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    self._sleep(
                        min(self.retry_base_seconds**attempt, RETRY_MAX_SECONDS)
                    )
            except PROVIDER_ERRORS as exc:
                raise ProviderError(
                    f"{name} rejected the request: {type(exc).__name__}: {exc}"
                ) from exc
        raise ProviderError(
            f"{name} failed after {self.max_retries} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def call(
        self,
        name: str,
        *,
        user: str,
        system: str = "",
        images: Sequence[str | Path] = (),
        max_tokens: int = 800,
        temperature: float = 0.0,
        reasoning: bool = True,
    ) -> GenResult:
        """Send one chat request and return the answer with its cost metadata.

        ``reasoning=False`` asks OpenRouter to switch off a model's hidden
        reasoning; ``True`` leaves the model's default alone.
        """
        spec = self.spec(name)
        price = self.pricing.price(name)
        image_paths = [Path(path) for path in images]
        if image_paths and not spec.vision:
            raise ConfigurationError(f"model {name} does not accept image input")
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"image input does not exist: {missing}")
        encoded = [encode_image(path) for path in image_paths]

        if spec.sdk == "anthropic":
            request = self._anthropic_request(
                spec, price, system, user, encoded, max_tokens, temperature
            )
        else:
            request = self._openai_request(
                spec, price, system, user, encoded, max_tokens, temperature, reasoning
            )
        return self._with_retries(name, request)

    def _anthropic_request(
        self,
        spec: ModelSpec,
        price: ModelPrice,
        system: str,
        user: str,
        encoded_images: Sequence[str],
        max_tokens: int,
        temperature: float,
    ) -> Callable[[], GenResult]:
        content: list[dict[str, object]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": data,
                },
            }
            for data in encoded_images
        ]
        content.append({"type": "text", "text": user})
        kwargs: dict[str, object] = {
            "model": spec.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system

        def request() -> GenResult:
            started = time.perf_counter()
            response = self._provider(spec.name).messages.create(**kwargs)
            latency = (time.perf_counter() - started) * 1000
            text = response.content[0].text if response.content else ""
            input_tokens = int(response.usage.input_tokens)
            output_tokens = int(response.usage.output_tokens)
            return GenResult(
                text=text.strip(),
                model=spec.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                cost_usd=price.cost(input_tokens, output_tokens),
            )

        return request

    def _openai_request(
        self,
        spec: ModelSpec,
        price: ModelPrice,
        system: str,
        user: str,
        encoded_images: Sequence[str],
        max_tokens: int,
        temperature: float,
        reasoning: bool = True,
    ) -> Callable[[], GenResult]:
        if encoded_images:
            content: object = [{"type": "text", "text": user}] + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{data}"},
                }
                for data in encoded_images
            ]
        else:
            content = user
        messages: list[dict[str, object]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        body: dict[str, object] = {}
        if spec.providers:
            body["provider"] = {"order": list(spec.providers), "allow_fallbacks": False}
        if not reasoning:
            body["reasoning"] = {"enabled": False}
        extra: dict[str, object] = {"extra_body": body} if body else {}

        def request() -> GenResult:
            started = time.perf_counter()
            response = self._provider(spec.name).chat.completions.create(
                model=spec.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                **extra,
            )
            latency = (time.perf_counter() - started) * 1000
            if not response.choices:
                extra_fields = getattr(response, "model_extra", None) or {}
                raise EmptyResponseError(
                    f"{spec.name} returned no choices; "
                    f"error={getattr(response, 'error', None) or extra_fields.get('error')}"
                )
            text = response.choices[0].message.content or ""
            usage = response.usage
            if usage is None:
                raise InvalidModelOutputError(
                    f"{spec.name} returned no token usage; cost cannot be attributed"
                )
            input_tokens = int(usage.prompt_tokens or 0)
            output_tokens = int(usage.completion_tokens or 0)
            return GenResult(
                text=text.strip(),
                model=spec.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                cost_usd=price.cost(input_tokens, output_tokens),
            )

        return request

    def judge(
        self, name: str, prompt: str, images: Sequence[str | Path] = ()
    ) -> tuple[int, GenResult]:
        """Binary judgement given after a justification, on a line of its own.

        A reply with no verdict line (in practice a reply cut off after a few
        words) is asked again, up to ``JUDGE_ATTEMPTS`` times in all; once a
        reply has used the whole token cap, later attempts run with reasoning
        off. The verdict always comes from one complete reply, and a reply that
        is malformed every time still stops the run. Discarded attempts stay in
        the cost.
        """
        spent = 0.0
        reasoning = True
        for attempt in range(1, JUDGE_ATTEMPTS + 1):
            result = self.call(
                name,
                user=prompt,
                images=images,
                max_tokens=JUDGE_MAX_TOKENS,
                reasoning=reasoning,
            )
            try:
                value = parse_verdict(name, result.text)
            except InvalidModelOutputError as exc:
                spent += result.cost_usd
                if result.output_tokens >= JUDGE_MAX_TOKENS:
                    reasoning = False
                if attempt == JUDGE_ATTEMPTS:
                    raise InvalidModelOutputError(
                        f"{exc} (last reply used {result.output_tokens} of "
                        f"{JUDGE_MAX_TOKENS} output tokens)"
                    ) from exc
                continue
            return value, replace(result, cost_usd=result.cost_usd + spent)
        raise AssertionError("unreachable")

    def embed(
        self,
        texts: Sequence[str],
        *,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 1000,
    ) -> EmbeddingResult:
        """Embed texts in batches that stay under the provider request cap."""
        if not texts:
            raise ConfigurationError("embed() requires at least one text")
        if batch_size < 1:
            raise ConfigurationError("batch_size must be at least 1")
        spec = self.spec(model)
        if spec.access != "embedding":
            raise ConfigurationError(f"{model} is not an embedding model")
        price = self.pricing.price(model)
        vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])

            def request(batch: list[str] = batch) -> GenResult:
                started = time.perf_counter()
                response = self._provider(spec.name).embeddings.create(
                    model=spec.model, input=batch
                )
                latency = (time.perf_counter() - started) * 1000
                vectors.extend(item.embedding for item in response.data)
                tokens = int(response.usage.prompt_tokens)
                return GenResult(
                    text="",
                    model=spec.name,
                    input_tokens=tokens,
                    output_tokens=0,
                    latency_ms=latency,
                    cost_usd=price.cost(tokens, 0),
                )

            total_tokens += self._with_retries(model, request).input_tokens
        return EmbeddingResult(
            vectors=np.asarray(vectors, dtype="float32"),
            input_tokens=total_tokens,
            cost_usd=price.cost(total_tokens, 0),
        )


def parse_verdict(judge: str, text: str) -> int:
    """The 0 or 1 on the last non-empty line of a judge reply.

    The line may carry a ``Grade:`` or ``Verdict:`` label and stray markdown,
    but nothing else; any other final line is an error, never a guess.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    match = VERDICT_LINE.match(lines[-1]) if lines else None
    if match is None:
        raise InvalidModelOutputError(
            f"judge {judge} did not end with a line holding only 0 or 1: {text!r}"
        )
    return int(match.group(1))
