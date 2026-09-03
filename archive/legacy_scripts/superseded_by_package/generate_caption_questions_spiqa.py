"""
Generates the CAPTION-ANSWERABLE figure question set for SPIQA, matched per-crop
to the pixel-only set that generate_questions.py produced.

Protocol (the clean two-protocol design):
  * SAME author as the pixel set: Claude Haiku 4.5.
  * SAME crops: reads questions_spiqa_figures.json and uses exactly its sources.
  * DIFFERENT information: the author sees ONLY the GPT-4o pipeline caption
    (spiqa_images/captions.json) — never the crop, never the author caption.

Because author and crops are held fixed and only the information shown to the
author varies, any performance difference between the two sets isolates the
effect of question construction — the per-crop matched design the PubLayNet
sets approximated only at pool level.

Also reports the Table III diagnostics for BOTH sets:
  * answers appearing verbatim in the caption (expected: high for this set,
    zero for the pixel set)
  * mean answer length

Reuses the canonical validators (leaks_modality, has_orphan_reference) from
generate_questions.py, so the two figure sets pass identical checks.

Setup: .env needs ANTHROPIC_API_KEY (same as generate_questions.py).

Usage:
    python generate_caption_questions_spiqa.py
    python generate_caption_questions_spiqa.py --pixel-file questions_spiqa_figures.json
"""

import argparse
import json
import os
import re

from dotenv import load_dotenv
from generate_questions import (
    CLAUDE_MODEL,
    claude_client,
    has_orphan_reference,
    leaks_modality,
    load_existing,
    parse_json_array,
    save,
)

load_dotenv()

IMAGE_DIR = "spiqa_images"
PIXEL_FILE = os.path.join("data", "questions", "questions_spiqa_figures.json")
OUT_FILE = os.path.join("data", "questions", "questions_spiqa_figures_caption.json")


PROMPT = """Below is a one-sentence description of a visual region from a scientific paper.

Write ONE question answerable using ONLY the information in this description,
plus its short answer.

RULE 1 — SELF-CONTAINED AND SPECIFIC:
The question will be asked against a corpus of scientific papers. It must name
the specific method, quantity, variable, or comparison involved, so that it is
unambiguous. Never write "the study", "the results", "the data" as bare
references.

RULE 2 — DO NOT REVEAL WHERE THE ANSWER LIVES:
The question must NOT contain: figure, table, chart, image, graph, diagram,
plot, panel, shown, depicted, described, illustrated.

RULE 3 — GROUNDED:
The answer must be expressible from the description alone. Invent nothing.

If the description is too vague to support a specific question, reply with an
empty array: []

DESCRIPTION:
\"\"\"
{caption}
\"\"\"

Respond with ONLY a JSON array containing exactly one object, no markdown fences:
[{{"q": "...", "answer": "..."}}]"""


def _norm(s):
    return re.sub(r"[^a-z0-9.]+", " ", s.lower()).strip()


def answer_in_caption(answer, caption):
    a, c = _norm(answer), _norm(caption)
    return len(a) >= 2 and a in c


def gen_caption_question(client, caption):
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        temperature=0,
        messages=[{"role": "user", "content": PROMPT.format(caption=caption)}],
    )
    parsed = parse_json_array(resp.content[0].text)
    if not parsed or "q" not in parsed[0]:
        return None
    item = parsed[0]
    q = item["q"].strip()
    if leaks_modality(q) or has_orphan_reference(q):
        return None
    answer = str(item.get("answer", "")).strip()
    if not q or not answer:
        return None
    return {"q": q, "answer": answer}


def main(pixel_file):
    with open(os.path.join(IMAGE_DIR, "captions.json"), encoding="utf-8") as f:
        captions = json.load(f)
    pixel_qs = load_existing(pixel_file)
    if not pixel_qs:
        print(f"No pixel questions in {pixel_file} — run generate_questions.py first.")
        return

    crops = [item["source"] for item in pixel_qs]
    print(
        f"Generating caption-answerable questions for the SAME {len(crops)} "
        f"crops as {pixel_file}.\n"
    )

    client = claude_client()
    out = load_existing(OUT_FILE)  # resumable
    done = {item["source"] for item in out}

    for name in crops:
        if name in done:
            continue
        caption = captions.get(name)
        if not caption:
            print(f"[skip] {name}: no pipeline caption")
            continue
        qa = gen_caption_question(client, caption)
        if qa is None:
            print(f"[skip] {name}: vague caption or validator reject")
            continue
        out.append(
            {"q": qa["q"], "source": name, "answer": qa["answer"], "type": "figure"}
        )
        save(out, OUT_FILE)  # write after each item
        print(f"[ok] {name}\n     Q: {qa['q']}")

    # ── Table III diagnostics for both sets ──────────────────────────────────
    def diag(items, label):
        n = len(items)
        in_cap = sum(
            answer_in_caption(i["answer"], captions.get(i["source"], "")) for i in items
        )
        mean_len = sum(len(i["answer"]) for i in items) / n if n else 0
        print(
            f"  {label:<22} n={n:<3} answers-in-caption={in_cap}/{n} "
            f"mean-answer-len={mean_len:.0f}"
        )
        return in_cap

    print(f"\nWrote {len(out)} caption questions -> {OUT_FILE}")
    print("\nTable III diagnostics:")
    diag(out, "caption-answerable")
    pix_in_cap = diag(pixel_qs, "pixel-only")
    if pix_in_cap:
        print(
            f"  WARNING: {pix_in_cap} pixel-only answers appear in their captions. "
            f"Replace those questions or the pixel-only property of Table III fails."
        )
    matched = {i["source"] for i in out} & set(crops)
    print(f"  crops with both protocols: {len(matched)}")
    print("\nSkim both files to 35, keeping the SAME crops in each where possible.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Caption-answerable figure questions matched per-crop to the pixel set."
    )
    ap.add_argument("--pixel-file", default=PIXEL_FILE)
    args = ap.parse_args()
    main(args.pixel_file)
