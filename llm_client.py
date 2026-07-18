"""
llm_client.py — one seam for every model call in the experiment.

Why this exists
───────────────
The four generate_* functions each built their own OpenAI client and hardcoded
CHAT_MODEL = "gpt-4o". That left nowhere to (a) swap the generator per run, or
(b) capture the tokens and latency the paper's efficiency tables need — the
functions returned a bare string.

This module is that seam. Every call now returns a GenResult carrying the text
AND its cost/latency/token metadata, so compare_all.py can log per-call data
without each system re-implementing it.

Providers
─────────
All four generators plus DeepSeek speak the OpenAI protocol, so one client class
with different base_urls covers them. Anthropic does not, so it gets its own path.

  GENERATORS (the comparison)
    gpt4o-mini         OpenAI     closed   vision
    gemini-flash-lite  Google     closed   vision   ← thinking model, see note
    llama4-maverick    DeepInfra  open     vision   ← FP8 serve
    llama4-scout       DeepInfra  open     vision

  Two closed, two open, all four with vision. The open pair moved off Groq
  (free-tier rate limits killed the pixel paths) and llama-3.3-70b was replaced
  by Llama 4 Maverick so that every generator can attempt figure questions.

  JUDGES (outside the generator set — no self-grading)
    deepseek           DeepSeek  accuracy + relevancy, and text faithfulness
    claude-haiku       Anthropic figure faithfulness (needs to SEE the figure)

Note on Gemini thinking tokens
──────────────────────────────
Gemini 2.5/3.x spend tokens on internal reasoning before emitting visible text,
and those count as output_tokens. We deliberately leave thinking ON: a
practitioner deploying Gemini pays for them, so it is the decision-relevant
number. Gemini's token/cost figures are therefore not strictly like-for-like
with the non-thinking models, and the paper footnotes this.

Pricing: USD per 1M tokens, current as of the experiment. Update PRICING if
providers change rates — cost_usd is computed, never returned by the API.
"""

import os
import io
import time
import base64
from dataclasses import dataclass, field

from PIL import Image
from dotenv import load_dotenv

load_dotenv()


# ── Result type ───────────────────────────────────────────────────────────────
@dataclass
class GenResult:
    """One model call: the answer plus everything the efficiency tables need."""
    text:       str
    model:      str
    in_tok:     int   = 0
    out_tok:    int   = 0
    latency_ms: float = 0.0
    cost_usd:   float = 0.0
    error:      str   = ""      # non-empty if the call failed after all retries


# ── Registry ──────────────────────────────────────────────────────────────────
# base_url None -> native OpenAI endpoint. "sdk" picks the client path.
MODELS = {
    # ---- generators ----
    "gpt4o-mini": {
        "sdk": "openai", "model": "gpt-4o-mini", "key": "OPENAI_API_KEY",
        "base_url": None, "vision": True,
        "access": "closed", "vendor": "OpenAI",
    },
    "gemini-flash-lite": {
        "sdk": "openai", "model": "gemini-3.1-flash-lite", "key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "vision": True, "access": "closed", "vendor": "Google",
    },
    # Both open models moved off Groq to DeepInfra.
    #
    # Groq's free tier rate-limited the pixel paths specifically: +multimodal and
    # +both carry a ~1,000-token base64 crop, so they exhaust a tokens-per-minute
    # bucket while baseline/+KG sail through. llm_client returns
    # GenResult(text="", error=...) after retries, the judges score "" as 0, and
    # the run lands a summary full of zeros clustered on exactly the two systems
    # the paper is about. A rate limit wearing the costume of a finding.
    # Groq's Developer tier was unavailable ("temporarily unavailable due to high
    # demand"), so the fix was a different host for the same open weights.
    #
    # These are the SAME MODELS Groq serves — open weights, multiple hosts. The
    # open/closed axis of the comparison is unchanged by the move.
    "llama4-maverick": {
        # Replaces llama-3.3-70b-versatile, which was TEXT-ONLY. On the
        # pixel-only figure set a text-only model can only score 0.000 — a
        # designed floor, not a measurement. Maverick has native vision, so all
        # four generators now actually attempt every question type and the figure
        # table is a real 4x4 grid.
        #
        # NOTE: this is the FP8-quantised serve, not bf16. Cheaper and faster,
        # but not numerically identical to the reference weights. The paper
        # footnotes it.
        "sdk": "openai", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "key": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "vision": True, "access": "open", "vendor": "Meta/DeepInfra",
    },
    "llama4-scout": {
        "sdk": "openai", "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "key": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "vision": True, "access": "open", "vendor": "Meta/DeepInfra",
    },

    # ---- diagnostic only, NOT part of the comparison ----
    # GPT-4o was the generator in the original 100-page report (figure accuracy
    # 0.514). GPT-4o-mini scores 0.086 on the 1,000-page corpus. Running this
    # isolates "the cheap model cannot read dense figures" from "something else
    # regressed". Deliberately absent from GENERATORS: at $2.50/$10.00 per 1M it
    # is not production-tier and must not drift into the headline comparison.
    "gpt4o": {
        "sdk": "openai", "model": "gpt-4o", "key": "OPENAI_API_KEY",
        "base_url": None, "vision": True,
        "access": "diagnostic", "vendor": "OpenAI",
    },

    # ---- judges ----
    "deepseek": {
        "sdk": "openai", "model": "deepseek-chat", "key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "vision": False, "access": "judge", "vendor": "DeepSeek",
    },
    "claude-haiku": {
        "sdk": "anthropic", "model": "claude-haiku-4-5", "key": "ANTHROPIC_API_KEY",
        "base_url": None, "vision": True,
        "access": "judge", "vendor": "Anthropic",
    },
}

GENERATORS = ["gpt4o-mini", "gemini-flash-lite", "llama4-maverick", "llama4-scout"]
# Selectable via --model for diagnostics, but never part of the reported
# comparison. Keeping these lists separate is what stops a ceiling check from
# quietly becoming a fifth generator in the results table.
DIAGNOSTICS = ["gpt4o"]

# USD per 1M tokens: (input, output)
PRICING = {
    # VERIFY these against DeepInfra's live pricing page before the final run —
    # they feed cost_usd -> cost_per_100q_usd, which is a headline table, and a
    # stale rate produces a wrong number that no test can catch.
    "gpt4o-mini":        (0.15, 0.60),
    "gemini-flash-lite": (0.10, 0.40),
    "llama4-maverick":   (0.15, 0.60),   # DeepInfra
    "llama4-scout":      (0.08, 0.30),   # DeepInfra (was 0.11/0.34 on Groq)
    "gpt4o":             (2.50, 10.00),   # diagnostic — 16x mini
    "deepseek":          (0.27, 1.10),
    "claude-haiku":      (1.00, 5.00),
}

MAX_RETRIES = 5
RETRY_BASE  = 2      # exponential backoff: 1, 2, 4, 8, 16 s


# ── Client construction (cached: one client per key) ──────────────────────────
_clients: dict = {}

def _client(name: str):
    if name in _clients:
        return _clients[name]
    cfg = MODELS[name]
    key = os.environ.get(cfg["key"])
    if not key:
        raise RuntimeError(f"{cfg['key']} missing from .env (needed for {name})")

    if cfg["sdk"] == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=key)
    else:
        from openai import OpenAI
        kwargs = {"api_key": key}
        if cfg["base_url"]:
            kwargs["base_url"] = cfg["base_url"]
        c = OpenAI(**kwargs)

    _clients[name] = c
    return c


def _cost(name: str, in_tok: int, out_tok: int) -> float:
    rate_in, rate_out = PRICING.get(name, (0.0, 0.0))
    return (in_tok / 1_000_000) * rate_in + (out_tok / 1_000_000) * rate_out


# ── Image encoding ────────────────────────────────────────────────────────────
def encode_image(path: str, max_px: int = 1200) -> str:
    """
    Base64 a crop, downscaling oversized ones.

    Crops come straight off the page at arbitrary sizes; providers reject very
    large images. 1200px is comfortably under every limit and preserves enough
    detail to read axis labels.
    """
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── The one call path ─────────────────────────────────────────────────────────
def call(name: str,
         system: str = "",
         user: str = "",
         images: list = None,
         max_tokens: int = 800,
         temperature: float = 0.0) -> GenResult:
    """
    Call any model in the registry. Retries transient failures with backoff.

    images: list of file paths. Silently ignored for text-only models — the
    caller is responsible for supplying captions instead (see the vision flag).

    Returns a GenResult even on total failure, with .error set, so a single bad
    call cannot kill a 1,200-call run.
    """
    cfg      = MODELS[name]
    images   = images or []
    use_imgs = images if cfg["vision"] else []

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()

            if cfg["sdk"] == "anthropic":
                content = []
                for p in use_imgs:
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png",
                                   "data": encode_image(p)},
                    })
                content.append({"type": "text", "text": user})
                kwargs = {
                    "model": cfg["model"],
                    "max_tokens": max_tokens,     # required by Anthropic
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": content}],
                }
                if system:
                    kwargs["system"] = system     # top-level, not a message
                resp = _client(name).messages.create(**kwargs)
                ms      = (time.perf_counter() - t0) * 1000
                text    = resp.content[0].text if resp.content else ""
                in_tok  = resp.usage.input_tokens
                out_tok = resp.usage.output_tokens

            else:
                if use_imgs:
                    content = [{"type": "text", "text": user}]
                    for p in use_imgs:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_image(p)}"
                            },
                        })
                else:
                    content = user

                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": content})

                resp = _client(name).chat.completions.create(
                    model=cfg["model"], temperature=temperature,
                    max_tokens=max_tokens, messages=messages,
                )
                ms   = (time.perf_counter() - t0) * 1000
                text = resp.choices[0].message.content or ""
                u    = resp.usage
                in_tok  = getattr(u, "prompt_tokens", 0) or 0
                out_tok = getattr(u, "completion_tokens", 0) or 0

            return GenResult(
                text=text.strip(), model=name,
                in_tok=in_tok, out_tok=out_tok, latency_ms=ms,
                cost_usd=_cost(name, in_tok, out_tok),
            )

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE ** attempt)

    # All retries exhausted: return a marked failure rather than raising, so the
    # run continues and the bad call is visible in the detail CSV.
    return GenResult(text="", model=name,
                     error=f"{type(last_err).__name__}: {str(last_err)[:120]}")


def judge(name: str, prompt: str, images: list = None) -> tuple:
    """
    Binary judge call. Returns (verdict, GenResult).

    Judges must emit a bare 1 or 0. We strip everything else and fail closed to
    0 on an unparseable reply, so a chatty judge degrades to a conservative
    verdict rather than a crash.
    """
    import re
    r = call(name, user=prompt, images=images, max_tokens=200)
    if r.error:
        return 0, r
    clean = re.sub(r"[^01]", "", r.text)
    return (1 if clean.startswith("1") else 0), r
