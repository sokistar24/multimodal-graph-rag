"""
test_clients.py — validate every API endpoint before running real experiments.

Checks, for each of the 6 roles in the experiment:
  • the key is present in .env
  • the client connects and authenticates
  • a real completion comes back
  • token usage is reported (needed for the cost/efficiency results)
  • latency is measurable

Roles tested
────────────
  GENERATORS (4)
    gpt4o-mini         OpenAI            closed
    gemini-flash-lite  Google            closed
    llama-70b          Groq (Meta)       open
    llama4-scout       Groq (Meta)       open
  JUDGE (1)
    deepseek           DeepSeek          binary 1/0 grading
  QUESTION-GEN (1)
    deepseek           DeepSeek          JSON output mode

Extra checks
────────────
  • Vision test  — sends a real image to each vision-capable generator.
                   Figure questions depend on this; a text-only endpoint
                   would silently wreck that whole question type.
  • JSON test    — question generation needs parseable JSON out of DeepSeek.
  • Judge test   — must return a bare "1" or "0", nothing else.

Setup:
    pip install openai google-genai python-dotenv pillow

    .env must contain:
        OPENAI_API_KEY=sk-...
        GEMINI_API_KEY=...
        GROQ_API_KEY=gsk-...
        DEEPSEEK_API_KEY=sk-...

Usage:
    python test_clients.py                     # test everything
    python test_clients.py --quick             # skip vision tests (faster/cheaper)
    python test_clients.py --only groq         # test one provider only
    python test_clients.py --list-models google  # what can my key actually reach?

Note on Gemini
──────────────
Gemini 2.5 models are "thinking" models: they consume tokens on internal
reasoning before producing visible output, and that reasoning counts against
max_tokens. Test calls therefore use MAX_TOKENS = 200 rather than a tight cap —
a 20-token cap truncates Gemini's reply mid-word and makes a healthy model look
broken. The cap is a ceiling, not a spend.

Each model may also declare "fallbacks": alternative IDs tried in order if the
primary fails, so a provider-side rename degrades to a warning rather than a
failed run. Use --list-models google to see what your key can actually reach.
"""

import os
import io
import re
import sys
import json
import time
import base64
import argparse

from PIL import Image, ImageDraw
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── ANSI colours for readable output ──────────────────────────────────────────
GREEN, RED, YELLOW, DIM, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
)
TICK, CROSS, WARN = f"{GREEN}✓{RESET}", f"{RED}✗{RESET}", f"{YELLOW}!{RESET}"

# ── Token cap for test calls ──────────────────────────────────────────────────
# 200, not 20. Gemini 2.5 models are "thinking" models: they spend tokens on
# internal reasoning before emitting visible output, and that reasoning counts
# against max_tokens. A 20-token cap truncated Gemini's reply to 'PI' (from
# 'PIPELINE_OK') and made a working model look broken. 200 gives thinking
# models headroom while keeping test spend negligible — this is a ceiling, not
# a spend, so non-thinking models still return their usual ~5 tokens.
#
# Note for later: this same effect will inflate Gemini's output-token counts in
# the real runs relative to non-thinking models. Whether to disable thinking for
# a fair cost/verbosity comparison is a compare_all.py decision, not this one.
MAX_TOKENS = 200


# ── Model registry ────────────────────────────────────────────────────────────
# base_url=None  → native OpenAI endpoint
# All four generators + DeepSeek are OpenAI-SDK compatible, which is why the
# whole experiment can use one client class with different base_urls.
MODELS = {
    "gpt4o-mini": {
        "provider": "OpenAI",
        "model":    "gpt-4o-mini",
        "key_env":  "OPENAI_API_KEY",
        "base_url": None,
        "role":     "generator (closed)",
        "vision":   True,
    },
    "gemini-flash-lite": {
        "provider": "Google",
        "model":    "gemini-3.1-flash-lite",
        # Confirmed present via --list-models. Fallbacks cover a rename or a
        # transient outage; not expected to be needed.
        "fallbacks": [
            "gemini-2.5-flash",
            "gemini-flash-lite-latest",
        ],
        "key_env":  "GEMINI_API_KEY",
        # Google exposes an OpenAI-compatible endpoint — keeps the code uniform.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "role":     "generator (closed)",
        "vision":   True,
    },
    "llama-70b": {
        "provider": "Groq",
        "model":    "llama-3.3-70b-versatile",
        "key_env":  "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "role":     "generator (open)",
        "vision":   False,   # text-only; uses caption-mediated variant
    },
    "llama4-scout": {
        "provider": "Groq",
        "model":    "meta-llama/llama-4-scout-17b-16e-instruct",
        "key_env":  "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "role":     "generator (open)",
        "vision":   True,
    },
    "deepseek": {
        "provider": "DeepSeek",
        "model":    "deepseek-chat",
        "key_env":  "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "role":     "judge + question-gen",
        "vision":   False,
    },
}


def make_client(cfg: dict) -> OpenAI:
    """Build an OpenAI-SDK client pointed at the right provider."""
    key = os.environ.get(cfg["key_env"])
    if not key:
        raise RuntimeError(f"{cfg['key_env']} missing from .env")
    kwargs = {"api_key": key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def list_available_models(cfg: dict) -> list:
    """
    Ask the provider which model IDs this key can actually reach.

    Providers rename and retire model IDs regularly (Gemini especially), so
    guessing from docs is unreliable — this asks the API directly.
    """
    client = make_client(cfg)
    try:
        return sorted(m.id for m in client.models.list())
    except Exception as e:
        print(f"  {WARN} could not list models: {str(e)[:60]}")
        return []


def resolve_model(cfg: dict) -> str | None:
    """
    Return the first model ID from cfg that actually responds.

    Tries cfg["model"], then each of cfg["fallbacks"]. Returns None if none
    work. This is why a Gemini rename doesn't break the whole run.
    """
    candidates = [cfg["model"]] + cfg.get("fallbacks", [])
    client = make_client(cfg)
    for model_id in candidates:
        try:
            client.chat.completions.create(
                model=model_id, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": "hi"}],
            )
            return model_id
        except Exception:
            continue
    return None


def make_test_image() -> str:
    """
    Build a tiny bar chart in memory and return it base64-encoded.

    Using a generated image rather than a file keeps the test self-contained.
    The chart is deliberately simple: a model that can see it should be able
    to say 'bar chart' or read the values.
    """
    img  = Image.new("RGB", (200, 150), "white")
    draw = ImageDraw.Draw(img)
    # Three bars of different heights
    draw.rectangle([30,  100, 60,  130], fill="steelblue")
    draw.rectangle([80,   60, 110, 130], fill="steelblue")
    draw.rectangle([130,  30, 160, 130], fill="steelblue")
    draw.line([20, 130, 180, 130], fill="black", width=2)   # x-axis
    draw.line([20,  20,  20, 130], fill="black", width=2)   # y-axis
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Individual tests ──────────────────────────────────────────────────────────
def test_text(name: str, cfg: dict) -> dict:
    """
    Basic text completion + token usage + latency.

    Tries cfg["model"] first; on failure walks cfg["fallbacks"]. The model ID
    that actually worked is returned in "resolved" so the caller can report it
    (and you can pin it in the registry afterwards).
    """
    client = make_client(cfg)
    candidates = [cfg["model"]] + cfg.get("fallbacks", [])
    last_err = None

    for model_id in candidates:
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model_id,
                temperature=0,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user",
                           "content": "Reply with exactly: PIPELINE_OK"}],
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            text  = resp.choices[0].message.content.strip()
            usage = resp.usage

            return {
                "ok":         "PIPELINE_OK" in text.upper(),
                "text":       text,
                "latency_ms": latency_ms,
                "resolved":   model_id,
                # Fell back to a non-primary name — worth surfacing.
                "fell_back":  model_id != cfg["model"],
                # Token counts are essential — cost/efficiency tables need them.
                "in_tok":     getattr(usage, "prompt_tokens", None),
                "out_tok":    getattr(usage, "completion_tokens", None),
            }
        except Exception as e:
            last_err = e
            continue

    raise last_err if last_err else RuntimeError("no candidates tried")


def test_vision(name: str, cfg: dict, model_id: str | None = None) -> dict:
    """
    Send a real image; confirms the figure-question path will work.

    model_id lets the caller pass the ID already resolved by test_text, so we
    don't re-discover it (and don't 404 on a stale primary name).
    """
    client = make_client(cfg)
    b64    = make_test_image()
    model_id = model_id or cfg["model"]

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        temperature=0,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "What kind of chart is this? Answer in 3 words or fewer."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = resp.choices[0].message.content.strip()

    # Pass if the model recognises it as a bar/column chart.
    ok = bool(re.search(r"\bbar\b|\bcolumn\b|\bchart\b|\bgraph\b", text, re.I))
    return {"ok": ok, "text": text, "latency_ms": latency_ms}


def test_judge(cfg: dict) -> dict:
    """
    The judge must return a bare 1 or 0.

    This mirrors exactly how compare_all.py will call it: a strict binary
    verdict with no prose. If the model wraps the digit in explanation, the
    parsing in the real run breaks — better to find out now.
    """
    client = make_client(cfg)
    prompt = (
        "You are grading an answer. Reply with ONLY the digit 1 or 0, "
        "no other text.\n\n"
        "Question: What is the capital of France?\n"
        "Reference answer: Paris\n"
        "Given answer: The capital of France is Paris.\n\n"
        "Is the given answer factually correct? Reply 1 or 0:"
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=cfg["model"], temperature=0, max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = resp.choices[0].message.content.strip()

    # Must be parseable as a bare 1 — and should be 1, since the answer is right.
    clean = re.sub(r"[^01]", "", text)
    return {
        "ok":         clean == "1",
        "text":       text,
        "latency_ms": latency_ms,
        "parsed":     clean,
    }


def test_json(cfg: dict) -> dict:
    """
    Question generation needs valid JSON out of DeepSeek.

    Tests the exact shape the question-gen script will request: a list of
    objects with question / answer / source fields.
    """
    client = make_client(cfg)
    prompt = (
        "Generate exactly 2 question-answer pairs about the solar system.\n"
        "Respond with ONLY a JSON array, no markdown fences, no preamble.\n"
        'Format: [{"question": "...", "answer": "...", "source": "..."}]'
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=cfg["model"], temperature=0, max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = resp.choices[0].message.content.strip()

    # Strip markdown fences if the model added them despite instructions.
    cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        parsed = json.loads(cleaned)
        ok = (isinstance(parsed, list)
              and len(parsed) >= 1
              and "question" in parsed[0])
    except Exception:
        parsed, ok = None, False

    return {
        "ok":         ok,
        "text":       text[:100],
        "latency_ms": latency_ms,
        "n_items":    len(parsed) if isinstance(parsed, list) else 0,
    }


# ── Runner ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Validate all experiment APIs.")
    ap.add_argument("--quick", action="store_true",
                    help="skip vision tests (faster, cheaper)")
    ap.add_argument("--only",  type=str, default=None,
                    help="test one provider only: openai|google|groq|deepseek")
    ap.add_argument("--list-models", type=str, default=None, metavar="PROVIDER",
                    help="list model IDs your key can reach, then exit. "
                         "e.g. --list-models google")
    args = ap.parse_args()

    # ── Discovery mode: ask a provider what it actually offers ───────────────
    # Useful when a model ID 404s and you need the current name.
    if args.list_models:
        target = args.list_models.lower()
        matches = [(n, c) for n, c in MODELS.items()
                   if target in c["provider"].lower() or target in n.lower()]
        if not matches:
            print(f"{RED}No provider matching {target!r}. "
                  f"Try: openai | google | groq | deepseek{RESET}")
            return 1
        name, cfg = matches[0]
        print(f"\nModels reachable with {cfg['key_env']} "
              f"({cfg['provider']}):\n")
        ids = list_available_models(cfg)
        if not ids:
            print(f"  {DIM}(none returned){RESET}")
            return 1
        for mid in ids:
            marker = f"  {GREEN}← configured{RESET}" if mid == cfg["model"] else ""
            print(f"  {mid}{marker}")
        print(f"\n{DIM}Pin your choice in the MODELS registry "
              f"at the top of this file.{RESET}\n")
        return 0

    print(f"\n{'=' * 68}")
    print("  API VALIDATION — multimodal graph-RAG experiment")
    print(f"{'=' * 68}")

    # ── Step 1: keys present? ────────────────────────────────────────────────
    print(f"\n{DIM}Step 1 — checking .env keys{RESET}")
    required = ["OPENAI_API_KEY", "GEMINI_API_KEY",
                "GROQ_API_KEY", "DEEPSEEK_API_KEY"]
    missing = []
    for k in required:
        val = os.environ.get(k)
        if val:
            print(f"  {TICK} {k:<20} {DIM}…{val[-4:]}{RESET}")
        else:
            print(f"  {CROSS} {k:<20} {RED}MISSING{RESET}")
            missing.append(k)

    if missing:
        print(f"\n{RED}Add these to .env before continuing:{RESET}")
        for k in missing:
            print(f"    {k}=...")
        return 1

    # ── Step 2: text completion for every model ──────────────────────────────
    print(f"\n{DIM}Step 2 — text completion + token usage + latency{RESET}")
    results  = {}
    failures = []

    for name, cfg in MODELS.items():
        if args.only and args.only.lower() not in cfg["provider"].lower():
            continue
        try:
            r = test_text(name, cfg)
            results[name] = r
            if r["ok"]:
                # Token counts must be present or the cost tables can't be built.
                tok = (f"{r['in_tok']}→{r['out_tok']} tok"
                       if r["in_tok"] is not None else f"{YELLOW}no usage data{RESET}")
                print(f"  {TICK} {name:<20} {cfg['provider']:<10} "
                      f"{r['latency_ms']:>6.0f}ms  {DIM}{tok}{RESET}")
                if r.get("fell_back"):
                    print(f"      {WARN} primary {cfg['model']!r} unavailable; "
                          f"using {GREEN}{r['resolved']!r}{RESET}")
                    print(f"      {DIM}→ pin this in MODELS to skip the "
                          f"fallback next time{RESET}")
                if r["in_tok"] is None:
                    print(f"      {WARN} token usage missing — cost metrics "
                          f"will not work for this model")
            else:
                print(f"  {CROSS} {name:<20} unexpected reply: {r['text'][:40]!r}")
                failures.append(name)
        except Exception as e:
            print(f"  {CROSS} {name:<20} {RED}{type(e).__name__}: "
                  f"{str(e)[:60]}{RESET}")
            failures.append(name)

    # ── Step 3: vision (figure questions depend on this) ─────────────────────
    if not args.quick:
        print(f"\n{DIM}Step 3 — vision (required for figure questions){RESET}")
        for name, cfg in MODELS.items():
            if args.only and args.only.lower() not in cfg["provider"].lower():
                continue
            if not cfg["vision"]:
                print(f"  {DIM}– {name:<20} text-only by design "
                      f"(caption-mediated variant){RESET}")
                continue
            try:
                # Reuse the ID that test_text proved works.
                resolved = results.get(name, {}).get("resolved")
                r = test_vision(name, cfg, model_id=resolved)
                if r["ok"]:
                    print(f"  {TICK} {name:<20} {r['latency_ms']:>6.0f}ms  "
                          f"{DIM}saw: {r['text'][:30]!r}{RESET}")
                else:
                    print(f"  {WARN} {name:<20} replied {r['text'][:40]!r} "
                          f"{DIM}(may still be fine){RESET}")
            except Exception as e:
                print(f"  {CROSS} {name:<20} {RED}vision failed: "
                      f"{str(e)[:50]}{RESET}")
                failures.append(f"{name}-vision")
    else:
        print(f"\n{DIM}Step 3 — vision  [skipped: --quick]{RESET}")

    # ── Step 4: judge binary output ──────────────────────────────────────────
    print(f"\n{DIM}Step 4 — DeepSeek judge (must return bare 1/0){RESET}")
    if not args.only or "deepseek" in (args.only or "").lower():
        try:
            r = test_judge(MODELS["deepseek"])
            if r["ok"]:
                print(f"  {TICK} judge returns clean binary  "
                      f"{r['latency_ms']:>6.0f}ms  {DIM}parsed: {r['parsed']!r}{RESET}")
            else:
                print(f"  {CROSS} judge returned {r['text'][:40]!r} "
                      f"→ parsed {r['parsed']!r} {RED}(expected '1'){RESET}")
                print(f"      {WARN} judge prompt may need tightening")
                failures.append("judge")
        except Exception as e:
            print(f"  {CROSS} judge test failed: {RED}{str(e)[:60]}{RESET}")
            failures.append("judge")

    # ── Step 5: JSON mode for question generation ────────────────────────────
    print(f"\n{DIM}Step 5 — DeepSeek JSON output (question generation){RESET}")
    if not args.only or "deepseek" in (args.only or "").lower():
        try:
            r = test_json(MODELS["deepseek"])
            if r["ok"]:
                print(f"  {TICK} valid JSON, {r['n_items']} items  "
                      f"{r['latency_ms']:>6.0f}ms")
            else:
                print(f"  {CROSS} JSON parse failed: {r['text'][:50]!r}")
                failures.append("json")
        except Exception as e:
            print(f"  {CROSS} JSON test failed: {RED}{str(e)[:60]}{RESET}")
            failures.append("json")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    if failures:
        print(f"{RED}  FAILED: {', '.join(failures)}{RESET}")
        print(f"{'=' * 68}")
        print("\nFix the above before running the pilot.")
        return 1

    print(f"{GREEN}  ALL CHECKS PASSED — ready for ingestion and the pilot run{RESET}")
    print(f"{'=' * 68}")

    # Latency preview: an early look at one of the paper's secondary results.
    if results:
        print(f"\n{DIM}Latency preview (single call — indicative only):{RESET}")
        for name, r in sorted(results.items(), key=lambda x: x[1]["latency_ms"]):
            role = MODELS[name]["role"]
            print(f"  {r['latency_ms']:>7.0f}ms  {name:<20} {DIM}{role}{RESET}")

    print(f"\n{DIM}Next: python ingest_publaynet.py --limit 5 --smoke{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())