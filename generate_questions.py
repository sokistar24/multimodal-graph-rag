"""
generate_questions.py — Phase 2. Build the 100-question evaluation set.

Split (per the locked design):
    35  text      — answerable from a single passage        → DeepSeek
    30  multi-hop — needs 2+ facts from the same page       → DeepSeek
    35  figure    — answerable ONLY from a figure/table     → Claude Haiku 4.5

Why two authors
───────────────
Both sit OUTSIDE the four generators under test (OpenAI / Google / Meta), so no
model grades or authors questions for its own family. DeepSeek is text-only,
which is fine for text and multi-hop; figure questions need vision, so Claude
reads the actual crop rather than a caption.

Output — matches the existing schema exactly, so compare_all.py reads it unchanged:
    questions_publaynet_text.json      [{"q","source","answer","type":"text"}]
    questions_publaynet_figures.json   [{"q","source","answer","type":"figure"}]
    questions_publaynet_multihop.json  [{"q","source","answer","type":"multihop"}]

`source` is the retrieval ground truth:
    .txt → scored against the text ranking
    .png → scored against the image ranking  (compare_all keys off this extension)

Protocol
────────
  • temperature 0, fixed seed → reproducible page sampling
  • SELF-CONTAINED: questions must name the specific entity involved, so that
    exactly one page in a 1,000-page corpus can answer them. "How many people
    attended the workshop?" is rejected — which workshop? Dozens of pages could
    answer, so there is no determinate ground truth AND retrieval has no
    distinctive terms to match on. Enforced by has_orphan_reference().
  • NO MODALITY LEAK: questions must not reveal where the answer lives
    ("as shown in the figure…" would tell the model to look at an image).
    Enforced by leaks_modality().
  • self-verification: every figure question is re-checked against its own crop
    by a second Claude call. This catches question↔image drift, which would
    silently corrupt the ground truth.
  • resumable: writes after each item; re-run to top up.
  • ALL ITEMS STILL NEED MANUAL REVIEW before the pilot.

Setup:
    pip install openai anthropic python-dotenv pillow
    .env needs: DEEPSEEK_API_KEY, ANTHROPIC_API_KEY

Usage:
    python generate_questions.py --smoke                 # 2 per type, inspect
    python generate_questions.py                         # full 35/30/35
    python generate_questions.py --only figure           # regenerate one type
    python generate_questions.py --corpus-dir publaynet_corpus \
                                 --image-dir publaynet_images \
                                 --prefix questions_publaynet
"""

import os
import io
import re
import sys
import json
import glob
import time
import base64
import random
import argparse

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN, RED, YELLOW, DIM, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
)
TICK, CROSS, WARN = f"{GREEN}✓{RESET}", f"{RED}✗{RESET}", f"{YELLOW}!{RESET}"

# ── Config ────────────────────────────────────────────────────────────────────
DEEPSEEK_MODEL = "deepseek-chat"
CLAUDE_MODEL   = "claude-haiku-4-5"
SEED           = 42
MAX_RETRIES    = 3

TARGETS = {"text": 35, "multihop": 30, "figure": 35}

# Words that would give away which modality the answer lives in. A question
# containing these tells the model where to look, which defeats the whole point
# of comparing text-only vs multimodal retrieval.
LEAK_PATTERN = re.compile(
    r"\b(figure|fig\.?|chart|image|graph|table|diagram|plot|panel|"
    r"shown|depicted|illustrat|pictured|above|below|visual)\b", re.I
)

# ── Orphan-reference detector ─────────────────────────────────────────────────
# A question like "How many people attended the workshop?" is grammatically
# self-contained but refers to a document the reader cannot identify. Over a
# 1,000-page corpus, dozens of pages could answer it, so there is no determinate
# ground truth AND retrieval cannot succeed: the question carries no distinctive
# content words for FAISS/CLIP to match. Such questions add noise to every
# generator equally and widen the confidence intervals.
#
# We reject a question when it leans on a bare definite reference ("the study",
# "this trial", "the current paper") without supplying identifying detail.
ORPHAN_PATTERN = re.compile(
    r"\b(?:the|this|these|those|that)\s+"
    # Allow up to two intervening modifiers: "the in-depth interviews",
    # "the current paper", "the present randomised trial".
    r"(?:[\w-]+\s+){0,2}"
    r"(stud(?:y|ies)|paper|article|workshop|trial|survey|experiment|"
    r"interviews?|questionnaires?|participants?|patients?|subjects?|"
    r"cohort|sample|authors?|researchers?|analysis|research|"
    r"investigation|project|programme|program|intervention|"
    r"dataset|data\s?set|model|method|approach|framework|"
    r"work|review|report|findings?|results?)\b",
    re.I,
)

# Signals that a question IS anchored to a specific entity despite mentioning a
# generic noun. A capitalised proper noun, a number, a year, or a unit gives
# FAISS something to retrieve on, so we don't reject on the generic noun alone.
ANCHOR_PATTERN = re.compile(
    r"[A-Z][a-z]{2,}"                 # a proper noun (Geneva, Malaria, WHO)
    r"|\b\d{4}\b"                     # a year
    r"|\b\d+(?:\.\d+)?\s?"            # a number with a unit
    r"(?:%|mg|ml|kg|mm|cm|µg|nm|mmol|units?|years?|months?|weeks?|days?)\b"
)

# Pages shorter than this rarely contain a well-formed answerable fact.
MIN_PAGE_CHARS = 400
# Multi-hop needs enough material for two distinct facts.
MIN_MULTIHOP_CHARS = 900


# ── Clients ───────────────────────────────────────────────────────────────────
def deepseek_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing from .env")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def claude_client():
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic not installed. Run: pip install anthropic")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing from .env")
    return anthropic.Anthropic(api_key=key)


# ── Shared helpers ────────────────────────────────────────────────────────────
def parse_json_array(text: str):
    """Strip markdown fences models add despite instructions, then parse."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


def leaks_modality(question: str) -> bool:
    return bool(LEAK_PATTERN.search(question))


def has_orphan_reference(question: str) -> bool:
    """
    True if the question depends on a document the reader can't identify.

    "How many people attended the workshop?"          → True  (which workshop?)
    "What is the aim of the current paper?"           → True  (which paper?)
    "How many attended the 2011 WHO Geneva workshop?" → False (anchored)
    "What was the mean HbA1c in the metformin arm?"   → False (anchored)

    The rule: a bare definite reference is only acceptable when the question
    also carries an anchor — a proper noun, year, or measured quantity — that
    lets retrieval find the one right page.
    """
    if not ORPHAN_PATTERN.search(question):
        return False
    # A generic noun is fine if something else pins the question down.
    # Strip the leading question word so "What"/"How" don't count as anchors.
    body = re.sub(r"^\s*(what|how|which|when|where|who|why)\b", "",
                  question, flags=re.I)
    return not bool(ANCHOR_PATTERN.search(body))


def encode_crop(path: str, max_px: int = 1200) -> str:
    """Base64 a crop, downscaling oversized ones (Anthropic rejects large images)."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def save(items: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def load_existing(path: str) -> list:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


# ── Text + multi-hop: DeepSeek ────────────────────────────────────────────────
TEXT_FEWSHOT = """Examples of GOOD questions — each names a specific entity, so exactly one document can answer it:

{"q": "What method is used to assess the spatial consistency of ERP maps across subjects?", "answer": "the grand mean ERP map"}
{"q": "What is of paramount importance for the activation of p70S6K?", "answer": "the phosphorylation of Serine residue in 411 position"}
{"q": "What vaccine is currently the only one used in China for leptospirosis?", "answer": "multivalent inactivated vaccine"}

Examples of BAD questions — grammatically fine, but hundreds of papers could answer them:

{"q": "How many people attended the workshop?"}          <- WHICH workshop?
{"q": "What is the aim of the current paper?"}            <- WHICH paper?
{"q": "How many participants were in the study?"}         <- WHICH study?"""

MULTIHOP_FEWSHOT = """Examples of GOOD multi-hop questions — each needs TWO separate facts AND names specific entities:

{"q": "What is the odds ratio for headache in individuals with chronic pain, and what is it for back pain?", "answer": "1.83 (1.36-2.46) for chronic pain, 2.72 (1.73-4.29) for back pain"}
{"q": "Which treatment group showed the greatest reduction in HbA1c, and what dose did that group receive?", "answer": "the high-dose metformin group, receiving 2000 mg daily"}

Examples of BAD questions — hundreds of papers could answer these:

{"q": "How many participants were in the study, and what was their mean age?"}   <- WHICH study?
{"q": "What did the authors conclude, and how many patients were enrolled?"}     <- WHICH authors?"""


def gen_text_questions(client, page_text: str, page_id: str,
                       multihop: bool = False) -> list:
    """One question from one page. Returns [] on failure — caller retries."""
    kind = "multihop" if multihop else "text"

    if multihop:
        instruction = (
            "Write ONE question that requires combining TWO SEPARATE facts from "
            "different parts of the passage. A reader must find both to answer. "
            "Do not write a question answerable from a single sentence."
        )
        fewshot = MULTIHOP_FEWSHOT
    else:
        instruction = (
            "Write ONE question answerable from a SINGLE sentence or passage in "
            "the text below."
        )
        fewshot = TEXT_FEWSHOT

    prompt = f"""{instruction}

RULE 1 — SELF-CONTAINED AND SPECIFIC (most important):
The question will be asked against a corpus of 1,000 unrelated scientific pages.
It must name the specific condition, cohort, molecule, method, place, or entity
involved, so that exactly ONE page can answer it.
- NEVER write "the study", "the paper", "this trial", "the workshop",
  "the participants", "the authors", "the current research" or similar bare
  references. The reader has no idea which document you mean.
- Instead, name the actual subject: not "the participants in the study" but
  "post-stroke patients", not "the workshop" but "the WHO malaria surveillance
  workshop".
- Do NOT copy a whole sentence from the passage. Rephrase in your own words
  while keeping the specific entity names.

RULE 2 — DO NOT REVEAL WHERE THE ANSWER LIVES:
The question must NOT contain: figure, table, chart, image, graph, diagram,
shown, depicted, above, below.

RULE 3 — GROUNDED:
The answer must be short and directly supported by the passage. Invent nothing.

If the passage is too garbled or generic to support a specific question, reply
with an empty array: []

{fewshot}

PASSAGE (from page {page_id}):
\"\"\"
{page_text[:4000]}
\"\"\"

Respond with ONLY a JSON array containing exactly one object, no markdown fences:
[{{"q": "...", "answer": "..."}}]"""

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = parse_json_array(resp.choices[0].message.content)
        if not parsed or "q" not in parsed[0]:
            return []
        item = parsed[0]
        if leaks_modality(item["q"]):
            return []                      # rejected; caller retries another page
        if has_orphan_reference(item["q"]):
            return []                      # "the study" etc — unretrievable
        return [{
            "q":      item["q"].strip(),
            "source": f"{page_id}.txt",    # .txt → scored on the text ranking
            "answer": str(item.get("answer", "")).strip(),
            "type":   kind,
        }]
    except Exception as e:
        print(f"    {DIM}api error: {str(e)[:60]}{RESET}")
        return []


# ── Figure: Claude (vision) ───────────────────────────────────────────────────
def gen_figure_question(client, crop_path: str) -> list:
    """Claude looks at the actual crop and writes a question about it."""
    image_name = os.path.basename(crop_path)
    try:
        b64 = encode_crop(crop_path)
    except Exception as e:
        print(f"    {DIM}bad image {image_name}: {str(e)[:40]}{RESET}")
        return []

    prompt = """Look at this figure or table from a scientific paper.

Write ONE question that can ONLY be answered by looking at it — a fact that
lives in the visual content, not in surrounding prose.

RULE 1 — SELF-CONTAINED AND SPECIFIC (most important):
The question will be asked against a corpus of 1,000 unrelated scientific pages
and their figures. It must name the specific variable, group, condition, or
measurement involved, so exactly ONE image can answer it.
- NEVER write "the study", "the table", "this trial", "the participants",
  "the interviews" or similar bare references — the reader has no idea which
  document you mean, and hundreds of pages would match.
- Name the actual subject visible in the image: the variable on the axis, the
  named treatment arm, the specific cohort, the labelled group.
  BAD:  "How many female students were included in the in-depth interviews?"
  GOOD: "How many female nursing students were surveyed about needle-stick
         injury reporting?"
  BAD:  "What does it show about glucose levels?"
  GOOD: "What was the mean plasma glucose level at 120 minutes in the
         metformin arm?"

RULE 2 — DO NOT REVEAL WHERE THE ANSWER LIVES:
The question must NOT contain: figure, table, chart, image, graph, diagram,
panel, shown, depicted, above, below.

RULE 3 — GROUNDED:
The answer must be short, concrete, and readable directly from the visual.

If the content is too vague or unlabelled to support a specific, uniquely
identifiable question, reply with an empty array: []

Respond with ONLY a JSON array containing exactly one object, no markdown fences:
[{"q": "...", "answer": "..."}]"""

    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        parsed = parse_json_array(resp.content[0].text)
        if not parsed or "q" not in parsed[0]:
            return []
        item = parsed[0]
        if leaks_modality(item["q"]):
            return []
        if has_orphan_reference(item["q"]):
            return []                      # unretrievable over a large corpus
        return [{
            "q":      item["q"].strip(),
            "source": image_name,          # .png → scored on the image ranking
            "answer": str(item.get("answer", "")).strip(),
            "type":   "figure",
        }]
    except Exception as e:
        print(f"    {DIM}api error: {str(e)[:60]}{RESET}")
        return []


def verify_figure_question(client, crop_path: str, item: dict) -> bool:
    """
    Re-check the question against its own crop with a fresh call.

    This exists because of a real observation during endpoint testing: the same
    image produced a description of one figure and a question about a different
    one. If question and image drift apart, the ground truth is wrong and every
    figure metric downstream is meaningless. Cheap insurance.
    """
    try:
        b64 = encode_crop(crop_path)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=10,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": b64}},
                    {"type": "text", "text":
                     f"Question: {item['q']}\n"
                     f"Proposed answer: {item['answer']}\n\n"
                     "Can this question be answered from this image, and is the "
                     "proposed answer correct according to it?\n"
                     "Reply with ONLY the digit 1 (yes) or 0 (no):"},
                ],
            }],
        )
        verdict = re.sub(r"[^01]", "", resp.content[0].text)
        return verdict == "1"
    except Exception:
        return False      # fail closed: unverifiable items are dropped


# ── Source pools ──────────────────────────────────────────────────────────────
def load_pages(corpus_dir: str, min_chars: int) -> list:
    """Return [(page_id, text)] for pages with enough content, shuffled by SEED."""
    pages = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read().strip()
        except Exception:
            continue
        if len(text) >= min_chars:
            pages.append((os.path.basename(path)[:-4], text))
    random.Random(SEED).shuffle(pages)
    return pages


def load_crops(image_dir: str) -> list:
    """All figure/table crops, shuffled by SEED. Excludes captions.json etc."""
    crops = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    random.Random(SEED).shuffle(crops)
    return crops


# ── Generation loops ──────────────────────────────────────────────────────────
def build_text_type(kind: str, target: int, corpus_dir: str,
                    out_path: str, smoke: bool) -> None:
    client = deepseek_client()
    items  = load_existing(out_path)
    used   = {i["source"] for i in items}     # one question per page

    min_chars = MIN_MULTIHOP_CHARS if kind == "multihop" else MIN_PAGE_CHARS
    pages = load_pages(corpus_dir, min_chars)
    if not pages:
        print(f"  {CROSS} no pages ≥{min_chars} chars in {corpus_dir}/")
        return

    print(f"\n{DIM}{kind}: {len(items)}/{target} done, "
          f"{len(pages)} candidate pages{RESET}")

    attempts = 0
    n_rejected = 0
    for page_id, text in pages:
        if len(items) >= target:
            break
        if f"{page_id}.txt" in used:
            continue
        attempts += 1
        if attempts > target * 6:            # pages are noisy; don't loop forever
            print(f"  {WARN} stopping: {n_rejected} rejections in {attempts} "
                  f"attempts — prompt may need loosening")
            break

        new = gen_text_questions(client, text, page_id,
                                 multihop=(kind == "multihop"))
        if not new:
            n_rejected += 1
            continue

        items.extend(new)
        used.add(f"{page_id}.txt")
        save(items, out_path)                # checkpoint every item

        if smoke:
            print(f"  {TICK} {page_id}")
            print(f"      Q: {new[0]['q'][:80]}")
            print(f"      A: {new[0]['answer'][:80]}")
        elif len(items) % 5 == 0:
            rate = n_rejected / attempts if attempts else 0
            print(f"  [{len(items)}/{target}]  {DIM}rejected {n_rejected} "
                  f"({rate:.0%}){RESET}")

    print(f"  {TICK} {kind}: {len(items)} items -> {out_path}")
    if n_rejected:
        print(f"      {DIM}{n_rejected} rejected (vague / leaky / "
              f"orphan reference){RESET}")


def build_figure_type(target: int, image_dir: str,
                      out_path: str, smoke: bool) -> None:
    client = claude_client()
    items  = load_existing(out_path)
    used   = {i["source"] for i in items}

    crops = load_crops(image_dir)
    if not crops:
        print(f"  {CROSS} no crops in {image_dir}/")
        return

    print(f"\n{DIM}figure: {len(items)}/{target} done, "
          f"{len(crops)} candidate crops{RESET}")

    attempts = n_rejected = n_unverified = 0
    for crop_path in crops:
        if len(items) >= target:
            break
        if os.path.basename(crop_path) in used:
            continue
        attempts += 1
        if attempts > target * 6:
            print(f"  {WARN} stopping: {n_rejected + n_unverified} rejections "
                  f"in {attempts} attempts")
            break

        new = gen_figure_question(client, crop_path)
        if not new:
            n_rejected += 1
            continue

        # Verification pass — drop anything that doesn't check out.
        if not verify_figure_question(client, crop_path, new[0]):
            n_unverified += 1
            if smoke:
                print(f"  {WARN} {os.path.basename(crop_path)} "
                      f"failed verification, dropped")
            continue

        items.extend(new)
        used.add(new[0]["source"])
        save(items, out_path)

        if smoke:
            print(f"  {TICK} {new[0]['source']}")
            print(f"      Q: {new[0]['q'][:80]}")
            print(f"      A: {new[0]['answer'][:80]}")
        elif len(items) % 5 == 0:
            rate = (n_rejected + n_unverified) / attempts if attempts else 0
            print(f"  [{len(items)}/{target}]  {DIM}rejected {n_rejected}, "
                  f"unverified {n_unverified} ({rate:.0%}){RESET}")

    print(f"  {TICK} figure: {len(items)} items -> {out_path}")
    if n_rejected or n_unverified:
        print(f"      {DIM}rejected {n_rejected} (vague/leaky), "
              f"{n_unverified} failed verification{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 — generate question sets.")
    ap.add_argument("--corpus-dir", default="publaynet_corpus")
    ap.add_argument("--image-dir",  default="publaynet_images")
    ap.add_argument("--prefix",     default="questions_publaynet",
                    help="output filename prefix")
    ap.add_argument("--only", choices=["text", "multihop", "figure"],
                    help="generate one type only")
    ap.add_argument("--smoke", action="store_true",
                    help="2 per type, verbose — inspect before the full run")
    args = ap.parse_args()

    targets = {k: (2 if args.smoke else v) for k, v in TARGETS.items()}

    print(f"\n{'=' * 70}")
    print("  PHASE 2 — QUESTION GENERATION")
    print(f"{'=' * 70}")
    print(f"{DIM}  text/multi-hop → DeepSeek   figure → Claude Haiku 4.5{RESET}")
    print(f"{DIM}  both authors sit outside the four generators under test{RESET}")

    paths = {
        "text":     f"{args.prefix}_text.json",
        "multihop": f"{args.prefix}_multihop.json",
        "figure":   f"{args.prefix}_figures.json",
    }

    try:
        if args.only in (None, "text"):
            build_text_type("text", targets["text"], args.corpus_dir,
                            paths["text"], args.smoke)
        if args.only in (None, "multihop"):
            build_text_type("multihop", targets["multihop"], args.corpus_dir,
                            paths["multihop"], args.smoke)
        if args.only in (None, "figure"):
            build_figure_type(targets["figure"], args.image_dir,
                              paths["figure"], args.smoke)
    except RuntimeError as e:
        print(f"\n{CROSS} {RED}{e}{RESET}\n")
        return 1

    print(f"\n{'=' * 70}")
    total = sum(len(load_existing(p)) for p in paths.values())
    print(f"  {total} questions written")
    print(f"{'=' * 70}")
    print(f"\n{YELLOW}  MANUAL REVIEW REQUIRED before the pilot:{RESET}")
    print(f"{DIM}    • verify each reference answer against its source{RESET}")
    print(f"{DIM}    • confirm no question reveals its modality{RESET}")
    print(f"{DIM}    • confirm multi-hop items genuinely need two facts{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
