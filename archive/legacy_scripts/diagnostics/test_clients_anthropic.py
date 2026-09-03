"""
test_clients_anthropic.py — validate the Anthropic endpoint before Phase 2.

Claude Haiku 4.5 has two jobs in this experiment, both needing vision:

  1. QUESTION-GEN (figure)   — looks at a real figure crop and writes a question
                               + reference answer about it.  35 per dataset.
  2. JUDGE (faithfulness)    — looks at the figure and decides whether an answer
                               is actually supported by it.  DeepSeek can't do
                               this: it's text-only and cannot see the evidence.

Both roles sit outside the four generators (OpenAI / Google / Meta), so no
self-preference bias.

Why this is a separate file from test_clients.py
────────────────────────────────────────────────
Anthropic is NOT OpenAI-SDK compatible.  The other five endpoints all speak the
OpenAI protocol, which is why one client class with different base_urls covers
them.  Claude needs:
  • the `anthropic` package, not `openai`
  • max_tokens as a REQUIRED arg (not optional)
  • a different image block shape:
      OpenAI     {"type": "image_url", "image_url": {"url": "data:..."}}
      Anthropic  {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png",
                                              "data": "..."}}
  • usage under resp.usage.input_tokens / .output_tokens
    (OpenAI uses .prompt_tokens / .completion_tokens)

Tests run
─────────
  1. Key present in .env
  2. Text completion + token usage + latency
  3. Vision on a synthetic chart          (sanity: can it see at all?)
  4. Vision on a REAL crop from your corpus (the actual Phase 2 input)
  5. Question-gen role  — JSON out, from an image
  6. Faithfulness-judge role — bare 1/0, from an image

Tests 4-6 are the ones that matter: they exercise the exact shapes Phase 2 uses.

Setup:
    pip install anthropic pillow python-dotenv
    .env needs:  ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python test_clients_anthropic.py
    python test_clients_anthropic.py --quick        # skip the real-crop tests
    python test_clients_anthropic.py --image-dir publaynet_images
"""

import os
import io
import re
import sys
import glob
import json
import time
import base64
import random
import argparse

from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN, RED, YELLOW, DIM, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
)
TICK, CROSS, WARN = f"{GREEN}✓{RESET}", f"{RED}✗{RESET}", f"{YELLOW}!{RESET}"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL      = "claude-haiku-4-5"
KEY_ENV    = "ANTHROPIC_API_KEY"
MAX_TOKENS = 500          # required by the SDK; roomy enough for JSON output
IMAGE_DIR  = "publaynet_images"


def make_client():
    """Anthropic client. Import is local so a missing package gives a clear error."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic package not installed.  Run:  pip install anthropic"
        )
    key = os.environ.get(KEY_ENV)
    if not key:
        raise RuntimeError(f"{KEY_ENV} missing from .env")
    return anthropic.Anthropic(api_key=key)


def img_block(b64: str, media_type: str = "image/png") -> dict:
    """
    Anthropic's image block shape — deliberately different from OpenAI's.
    Getting this wrong is the single most likely Phase 2 bug, so it lives in
    one function that every test below shares.
    """
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64},
    }


def encode_image(path: str, max_px: int = 1200) -> tuple:
    """
    Read a crop from disk and base64 it.

    Downscales very large crops: Anthropic rejects oversized images, and your
    figure crops are arbitrary sizes straight from the page. Phase 2 will need
    this same guard, so it's tested here.
    """
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), img.size


def make_synthetic_chart() -> str:
    """A trivial bar chart — baseline check that vision works at all."""
    img  = Image.new("RGB", (200, 150), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 100,  60, 130], fill="steelblue")
    draw.rectangle([80,  60, 110, 130], fill="steelblue")
    draw.rectangle([130, 30, 160, 130], fill="steelblue")
    draw.line([20, 130, 180, 130], fill="black", width=2)
    draw.line([20,  20,  20, 130], fill="black", width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call(client, content, max_tokens: int = MAX_TOKENS) -> dict:
    """One Anthropic call. Returns text, tokens, latency — or the error."""
    try:
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,       # required by Anthropic, unlike OpenAI
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        ms = (time.perf_counter() - t0) * 1000
        return {
            "ok":      True,
            "text":    resp.content[0].text.strip(),
            "ms":      ms,
            # Note the different attribute names vs OpenAI.
            "in_tok":  resp.usage.input_tokens,
            "out_tok": resp.usage.output_tokens,
            "error":   None,
        }
    except Exception as e:
        return {"ok": False, "text": None, "ms": None,
                "in_tok": None, "out_tok": None,
                "error": f"{type(e).__name__}: {e}"}


def show_err(r: dict, indent: int = 6) -> None:
    """Print the full error — truncating errors is how we missed the Gemini 404."""
    pad = " " * indent
    for line in str(r["error"]).split("\n"):
        print(f"{pad}{RED}{line[:160]}{RESET}")


def pick_crops(image_dir: str, n: int = 2) -> list:
    """
    Grab a couple of real figure crops from the ingested corpus.

    Prefers 'figure' crops over 'table' — figures are the harder visual case
    and the one CLIP retrieval struggles with.
    """
    figures = glob.glob(os.path.join(image_dir, "*figure*.png"))
    tables  = glob.glob(os.path.join(image_dir, "*table*.png"))
    pool    = figures if figures else tables
    if not pool:
        pool = [p for p in glob.glob(os.path.join(image_dir, "*.png"))]
    if not pool:
        return []
    random.seed(42)                      # reproducible pick
    return random.sample(pool, min(n, len(pool)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate Anthropic for figure question-gen and judging.")
    ap.add_argument("--quick", action="store_true",
                    help="skip tests that use real corpus crops")
    ap.add_argument("--image-dir", type=str, default=IMAGE_DIR,
                    help=f"where the ingested crops live (default: {IMAGE_DIR})")
    args = ap.parse_args()

    print(f"\n{'=' * 70}")
    print("  ANTHROPIC VALIDATION — figure question-gen + faithfulness judge")
    print(f"{'=' * 70}")

    # ── 1. Key ───────────────────────────────────────────────────────────────
    print(f"\n{DIM}Step 1 — key{RESET}")
    key = os.environ.get(KEY_ENV)
    if not key:
        print(f"  {CROSS} {KEY_ENV:<20} {RED}MISSING{RESET}")
        print(f"\n{RED}Add to .env:{RESET}\n    {KEY_ENV}=sk-ant-...")
        return 1
    print(f"  {TICK} {KEY_ENV:<20} {DIM}…{key[-4:]}{RESET}")

    try:
        client = make_client()
    except RuntimeError as e:
        print(f"  {CROSS} {RED}{e}{RESET}")
        return 1

    failures = []

    # ── 2. Text ──────────────────────────────────────────────────────────────
    print(f"\n{DIM}Step 2 — text completion{RESET}")
    r = call(client, "Reply with exactly: PIPELINE_OK")
    if not r["ok"]:
        print(f"  {CROSS} text call failed")
        show_err(r)
        return 1
    if "PIPELINE_OK" in r["text"].upper():
        print(f"  {TICK} {MODEL:<22} {r['ms']:>6.0f}ms  "
              f"{DIM}{r['in_tok']}→{r['out_tok']} tok{RESET}")
    else:
        print(f"  {WARN} unexpected reply: {r['text'][:50]!r}")

    # ── 3. Synthetic vision ──────────────────────────────────────────────────
    print(f"\n{DIM}Step 3 — vision, synthetic chart{RESET}")
    r = call(client, [
        img_block(make_synthetic_chart()),
        {"type": "text", "text": "What kind of chart is this? 3 words max."},
    ])
    if not r["ok"]:
        print(f"  {CROSS} vision call failed")
        show_err(r)
        failures.append("vision")
    elif re.search(r"\bbar\b|\bcolumn\b|\bchart\b", r["text"], re.I):
        print(f"  {TICK} sees images  {r['ms']:>6.0f}ms  "
              f"{DIM}said: {r['text'][:35]!r}{RESET}")
    else:
        print(f"  {WARN} replied {r['text'][:50]!r}")

    # ── 4-6. Real crops: the tests that actually matter ─────────────────────
    if args.quick:
        print(f"\n{DIM}Steps 4-6 — real-crop tests  [skipped: --quick]{RESET}")
    else:
        crops = pick_crops(args.image_dir)
        if not crops:
            print(f"\n  {WARN} no crops found in {args.image_dir}/ — "
                  f"run ingestion first, or pass --image-dir")
        else:
            # ── 4. Read a real crop ──────────────────────────────────────────
            print(f"\n{DIM}Step 4 — vision on a REAL crop from your corpus{RESET}")
            crop_path = crops[0]
            b64, size = encode_image(crop_path)
            r = call(client, [
                img_block(b64),
                {"type": "text",
                 "text": "Describe this figure in one sentence."},
            ])
            name = os.path.basename(crop_path)
            if not r["ok"]:
                print(f"  {CROSS} failed on {name}")
                show_err(r)
                failures.append("real-crop")
            else:
                print(f"  {TICK} {name}  {DIM}{size[0]}×{size[1]}px  "
                      f"{r['ms']:.0f}ms  {r['in_tok']}→{r['out_tok']} tok{RESET}")
                print(f"      {DIM}{r['text'][:90]}{RESET}")

            # ── 5. Question-gen role ─────────────────────────────────────────
            # Exactly the shape Phase 2 uses: image in, strict JSON out.
            print(f"\n{DIM}Step 5 — question-gen role (image → JSON){RESET}")
            r = call(client, [
                img_block(b64),
                {"type": "text", "text":
                 "Write 1 question answerable ONLY by looking at this figure, "
                 "plus its reference answer.\n"
                 "The question must NOT mention the figure, chart, or image — "
                 "it should read as a plain factual question.\n"
                 "Respond with ONLY a JSON array, no markdown fences:\n"
                 '[{"question": "...", "answer": "...", "evidence": "..."}]'},
            ])
            if not r["ok"]:
                print(f"  {CROSS} question-gen failed")
                show_err(r)
                failures.append("question-gen")
            else:
                cleaned = re.sub(r"^```(?:json)?|```$", "", r["text"],
                                 flags=re.M).strip()
                try:
                    parsed = json.loads(cleaned)
                    ok = (isinstance(parsed, list) and len(parsed) >= 1
                          and "question" in parsed[0])
                    if ok:
                        q = parsed[0]["question"]
                        print(f"  {TICK} valid JSON  {r['ms']:>6.0f}ms")
                        print(f"      {DIM}Q: {q[:75]}{RESET}")
                        print(f"      {DIM}A: {str(parsed[0].get('answer'))[:75]}{RESET}")
                        # Protocol check: wording must not reveal the modality.
                        if re.search(r"\bfigure\b|\bchart\b|\bimage\b|\bgraph\b|"
                                     r"\btable\b|\bshown\b|\bdepict", q, re.I):
                            print(f"      {WARN} question leaks the modality — "
                                  f"prompt needs tightening in Phase 2")
                    else:
                        print(f"  {CROSS} JSON wrong shape: {cleaned[:60]!r}")
                        failures.append("question-gen")
                except Exception:
                    print(f"  {CROSS} not valid JSON: {r['text'][:60]!r}")
                    failures.append("question-gen")

            # ── 6. Faithfulness-judge role ───────────────────────────────────
            # The role DeepSeek cannot fill: verify an answer against an IMAGE.
            print(f"\n{DIM}Step 6 — faithfulness judge (image → bare 1/0){RESET}")

            # 6a. An answer that is obviously NOT supported → must return 0.
            r = call(client, [
                img_block(b64),
                {"type": "text", "text":
                 "You are grading whether an answer is supported by the figure "
                 "above.\nReply with ONLY the digit 1 or 0.\n\n"
                 "Answer given: 'The figure shows the annual rainfall in "
                 "Antarctica measured in millimetres.'\n\n"
                 "Is this answer supported by the figure? Reply 1 or 0:"},
                ], max_tokens=10)
            if not r["ok"]:
                print(f"  {CROSS} judge call failed")
                show_err(r)
                failures.append("judge")
            else:
                clean = re.sub(r"[^01]", "", r["text"])
                if clean == "0":
                    print(f"  {TICK} correctly rejected an unsupported answer  "
                          f"{DIM}{r['ms']:.0f}ms  parsed: {clean!r}{RESET}")
                elif clean == "1":
                    print(f"  {WARN} accepted a clearly wrong answer "
                          f"{DIM}(parsed {clean!r}) — judge prompt too lenient{RESET}")
                    failures.append("judge-lenient")
                else:
                    print(f"  {CROSS} unparseable verdict: {r['text'][:40]!r}")
                    failures.append("judge")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    if failures:
        print(f"{RED}  FAILED: {', '.join(failures)}{RESET}")
        print(f"{'=' * 70}\n")
        return 1
    print(f"{GREEN}  ANTHROPIC READY — figure question-gen + faithfulness judge{RESET}")
    print(f"{'=' * 70}")
    print(f"\n{DIM}All six roles now validated:{RESET}")
    print(f"{DIM}  4 generators + DeepSeek (text q-gen, judge) "
          f"+ Claude (figure q-gen, figure faithfulness){RESET}")
    print(f"\n{DIM}Next: Phase 2 — question generation{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
