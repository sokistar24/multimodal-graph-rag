"""Provider registry, frozen pricing, and the single model-call seam.

Every generator, judge, question author, captioner, triple extractor, and
embedding request in the project goes through :class:`ModelClient`. The client
records tokens, latency, and cost for each call from a pricing snapshot that is
loaded explicitly; a model without a frozen price cannot be charged, and a
provider failure that survives the retry budget is raised, never converted into
an empty answer.

Generators under comparison
    gpt4o-mini          OpenAI      closed  vision
    gemini-flash-lite   Google      closed  vision (thinking tokens are billed
                                            as output tokens and are left on)
    llama4-maverick     DeepInfra   open    vision (FP8 serve)
    llama4-scout        DeepInfra   open    vision

Diagnostic generator (never part of the reported comparison)
    gpt4o               OpenAI

Judges and question authors (outside every generator family)
    deepseek            DeepSeek    text judge and text question author
    claude-haiku        Anthropic   vision judge and figure question author

Embeddings
    text-embedding-3-small  OpenAI
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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


MODELS: dict[str, ModelSpec] = {
    "gpt4o-mini": ModelSpec(
        name="gpt4o-mini",
        sdk="openai",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        vendor="OpenAI",
        access="closed",
        vision=True,
    ),
    "gemini-flash-lite": ModelSpec(
        name="gemini-flash-lite",
        sdk="openai",
        model="gemini-3.1-flash-lite",
        api_key_env="GEMINI_API_KEY",
        vendor="Google",
        access="closed",
        vision=True,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "llama4-maverick": ModelSpec(
        name="llama4-maverick",
        sdk="openai",
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        api_key_env="DEEPINFRA_API_KEY",
        vendor="Meta/DeepInfra",
        access="open",
        vision=True,
        base_url="https://api.deepinfra.com/v1/openai",
    ),
    "llama4-scout": ModelSpec(
        name="llama4-scout",
        sdk="openai",
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        api_key_env="DEEPINFRA_API_KEY",
        vendor="Meta/DeepInfra",
        access="open",
        vision=True,
        base_url="https://api.deepinfra.com/v1/openai",
    ),
    "gpt4o": ModelSpec(
        name="gpt4o",
        sdk="openai",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        vendor="OpenAI",
        access="diagnostic",
        vision=True,
    ),
    "deepseek": ModelSpec(
        name="deepseek",
        sdk="openai",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        vendor="DeepSeek",
        access="judge",
        vision=False,
        base_url="https://api.deepseek.com",
    ),
    "claude-haiku": ModelSpec(
        name="claude-haiku",
        sdk="anthropic",
        model="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        vendor="Anthropic",
        access="judge",
        vision=True,
    ),
    "text-embedding-3-small": ModelSpec(
        name="text-embedding-3-small",
        sdk="openai",
        model="text-embedding-3-small",
        api_key_env="OPENAI_API_KEY",
        vendor="OpenAI",
        access="embedding",
        vision=False,
    ),
}

GENERATORS: tuple[str, ...] = (
    "gpt4o-mini",
    "gemini-flash-lite",
    "llama4-maverick",
    "llama4-scout",
)
DIAGNOSTICS: tuple[str, ...] = ("gpt4o",)
JUDGES: tuple[str, ...] = ("deepseek", "claude-haiku")
EMBEDDING_MODEL = "text-embedding-3-small"
CAPTION_MODEL = "gpt4o"
EXTRACTION_MODEL = "gpt4o-mini"

RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
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
        max_retries: int = 5,
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
                    self._sleep(self.retry_base_seconds**attempt)
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
    ) -> GenResult:
        """Send one chat request and return the answer with its cost metadata."""
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
                spec, price, system, user, encoded, max_tokens, temperature
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

        def request() -> GenResult:
            started = time.perf_counter()
            response = self._provider(spec.name).chat.completions.create(
                model=spec.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
            latency = (time.perf_counter() - started) * 1000
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
        """Binary judgement: the model must answer with exactly ``0`` or ``1``."""
        result = self.call(name, user=prompt, images=images, max_tokens=8)
        verdict = result.text.strip()
        if verdict not in {"0", "1"}:
            raise InvalidModelOutputError(
                f"judge {name} returned {verdict!r} instead of a bare 0 or 1"
            )
        return int(verdict), result

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
