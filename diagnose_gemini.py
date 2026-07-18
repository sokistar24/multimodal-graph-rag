"""
diagnose_gemini.py — one-off probe. Delete after use.

Question: why does `gemini-2.5-flash-lite` fail when --list-models says it exists,
while `gemini-2.5-flash` works on the same endpoint and key?

The probe prints the FULL error for each hypothesis rather than swallowing it,
so we diagnose from evidence instead of guessing.

Hypotheses tested
─────────────────
  H1  Name needs the `models/` prefix that the discovery endpoint returns.
  H2  Thinking tokens: model works but burns the whole budget before emitting
      visible text (this is what made Flash look broken at max_tokens=20).
  H3  The OpenAI-compat shim doesn't route this ID; native SDK does.
  H4  Model is listed but not enabled for chat completions on this key.

Then: benchmarks every cheap Gemini tier the key can reach, so we can pick a
replacement that is neither flash-lite (if broken) nor full flash (too pricey).

Usage:
    python diagnose_gemini.py
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
)
TICK, CROSS, WARN = f"{GREEN}✓{RESET}", f"{RED}✗{RESET}", f"{YELLOW}!{RESET}"

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
KEY      = os.environ.get("GEMINI_API_KEY")

client = OpenAI(api_key=KEY, base_url=BASE_URL)


def probe(model_id: str, max_tokens: int = 200, label: str = "") -> dict:
    """
    Single call. Returns everything we need to tell the failure modes apart:
    did it error, or did it succeed-but-emit-nothing (the thinking-token trap)?
    """
    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model_id,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": "Reply with exactly: PIPELINE_OK"}],
        )
        ms = (time.perf_counter() - t0) * 1000

        text  = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        # completion_tokens counts thinking + visible. If it's large but text is
        # empty/short, the budget went to reasoning — that's H2, not a failure.
        return {
            "ok":       "PIPELINE_OK" in text.upper(),
            "text":     text,
            "ms":       ms,
            "in_tok":   getattr(usage, "prompt_tokens", None),
            "out_tok":  getattr(usage, "completion_tokens", None),
            "finish":   resp.choices[0].finish_reason,
            "error":    None,
        }
    except Exception as e:
        return {
            "ok": False, "text": None, "ms": None,
            "in_tok": None, "out_tok": None, "finish": None,
            "error": f"{type(e).__name__}: {e}",
        }


def show(label: str, r: dict, indent: int = 2) -> None:
    pad = " " * indent
    if r["error"]:
        print(f"{pad}{CROSS} {label}")
        # Print the WHOLE error — the answer is usually in here.
        for line in str(r["error"]).split("\n"):
            print(f"{pad}   {RED}{line[:150]}{RESET}")
    elif r["ok"]:
        print(f"{pad}{TICK} {label}  {DIM}{r['ms']:.0f}ms  "
              f"{r['in_tok']}→{r['out_tok']} tok  finish={r['finish']}{RESET}")
    else:
        # Connected but didn't say what we asked — the interesting middle case.
        print(f"{pad}{WARN} {label}  {DIM}{r['ms']:.0f}ms  "
              f"{r['in_tok']}→{r['out_tok']} tok  finish={r['finish']}{RESET}")
        print(f"{pad}   replied: {r['text']!r}")


def main() -> None:
    if not KEY:
        print(f"{RED}GEMINI_API_KEY missing from .env{RESET}")
        return

    print(f"\n{'=' * 70}")
    print("  GEMINI DIAGNOSTIC — why does flash-lite fail?")
    print(f"{'=' * 70}")

    # ── H1: naming ────────────────────────────────────────────────────────────
    print(f"\n{DIM}H1 — is it the model name / prefix?{RESET}")
    print(f"{DIM}   discovery returns 'models/X'; compat layer may want bare 'X'{RESET}\n")
    for mid in [
        "gemini-2.5-flash-lite",           # bare — what we currently send
        "models/gemini-2.5-flash-lite",    # prefixed — as discovery lists it
        "gemini-flash-lite-latest",        # the moving alias
        "gemini-2.5-flash",                # known-good control
    ]:
        show(mid, probe(mid))

    # ── H2: thinking tokens ───────────────────────────────────────────────────
    print(f"\n{DIM}H2 — thinking tokens starving the visible reply?{RESET}")
    print(f"{DIM}   if out_tok is high but text is empty, reasoning ate the budget{RESET}\n")
    for mt in [20, 200, 2000]:
        r = probe("gemini-2.5-flash-lite", max_tokens=mt)
        show(f"flash-lite @ max_tokens={mt}", r)

    # ── H3: native SDK vs compat shim ────────────────────────────────────────
    print(f"\n{DIM}H3 — does the native google-genai SDK reach it?{RESET}")
    print(f"{DIM}   isolates 'model broken' from 'OpenAI-compat shim broken'{RESET}\n")
    try:
        from google import genai
        g = genai.Client(api_key=KEY)
        t0 = time.perf_counter()
        resp = g.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Reply with exactly: PIPELINE_OK",
        )
        ms = (time.perf_counter() - t0) * 1000
        txt = (resp.text or "").strip()
        if "PIPELINE_OK" in txt.upper():
            print(f"  {TICK} native SDK works  {DIM}{ms:.0f}ms  →  the model is "
                  f"fine; the OpenAI-compat shim is the problem{RESET}")
        else:
            print(f"  {WARN} native SDK replied {txt[:40]!r}")
    except ImportError:
        print(f"  {DIM}– google-genai not installed "
              f"(pip install google-genai) — skipping{RESET}")
    except Exception as e:
        print(f"  {CROSS} native SDK also fails: {RED}{str(e)[:100]}{RESET}")
        print(f"     {DIM}→ model itself is unavailable, not a shim issue{RESET}")

    # ── Alternatives: cheap tiers that are neither flash-lite nor full flash ──
    print(f"\n{DIM}Alternatives — cheap Gemini tiers on this key{RESET}")
    print(f"{DIM}   (from your --list-models output; excludes image/audio/tts){RESET}\n")
    for mid in [
        "gemini-2.0-flash-lite",       # older lite, very cheap
        "gemini-2.0-flash",            # older flash
        "gemini-3.1-flash-lite",       # newer lite tier
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
    ]:
        show(mid, probe(mid))

    print(f"\n{'=' * 70}")
    print("  Read the errors above, then pin the winner in test_clients.py")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
