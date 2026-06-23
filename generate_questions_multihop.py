"""
Generates multi-hop questions from the PubLayNet corpus.

A multi-hop question needs two distinct facts from different parts of the same page,
rather than a single lookup. Each page's full text is given to the model, which
returns one such question and its answer; the source is the page. The set is
over-generated and then manually filtered down to the genuinely two-fact ones.

Questions are tagged "type": "multihop".

Usage:
    python generate_questions_multihop.py --n 60
"""
import os
import glob
import json
import random
import argparse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

GEN_MODEL = "gpt-4o-mini"
CORPUS_DIR = "publaynet_corpus"
OUT_FILE = "questions_publaynet_multihop.json"

# Few-shot examples steer the model toward genuine two-fact questions rather than
# single lookups dressed up with an "and".
FEWSHOT = """Examples of GOOD multi-hop questions (each needs TWO separate facts):

Example 1:
{"q": "What method was used to measure the primary outcome, and what value did it produce?", "answer": "qPCR was used, producing a 2.3-fold increase"}

Example 2:
{"q": "Which patient group showed the highest expression, and which gene was measured?", "answer": "the treatment group showed the highest expression of Sox2"}

Notice each question combines TWO distinct facts that would typically appear in
different sentences or regions, not a single lookup.
"""


def generate_qa(page_text):
    """Returns one multi-hop {"q", "answer"} from a page, or None if the page won't
    support a genuine two-fact question (model replies SKIP)."""
    prompt = (
        FEWSHOT +
        "\nNow read the PAGE TEXT below (OCR'd from a scientific paper, may contain "
        "errors) and write ONE multi-hop question that requires combining TWO distinct "
        "facts from DIFFERENT parts of the text, plus its short answer.\n"
        "Rules:\n"
        "- The question MUST require two separate facts, not one lookup.\n"
        "- Both facts must be present in the text.\n"
        "- If the text doesn't support a genuine two-fact question, respond with "
        '{"q": "SKIP", "answer": "SKIP"}.\n'
        "- Respond with ONLY a JSON object: {\"q\": \"...\", \"answer\": \"...\"}\n\n"
        f"PAGE TEXT:\n{page_text[:3000]}"
    )
    resp = client.chat.completions.create(
        model=GEN_MODEL, temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        if result.get("q", "SKIP") == "SKIP":
            return None
        return {"q": result["q"], "answer": result["answer"]}
    except (json.JSONDecodeError, KeyError):
        return None


def main(n):
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    if not files:
        print(f"No .txt files in {CORPUS_DIR}/ -- run ingestion first.")
        return

    random.seed(7)                                      # fixed seed for reproducible sampling
    sampled = random.sample(files, min(n, len(files)))

    questions = []
    for path in sampled:
        page_key = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if len(text) < 400:                             # too short to hold two facts
            print(f"[skip] {page_key}: page too short")
            continue
        qa = generate_qa(text)
        if qa is None:
            print(f"[skip] {page_key}: no genuine two-fact question")
            continue
        questions.append({
            "q": qa["q"],
            "source": f"{page_key}.txt",
            "answer": qa["answer"],
            "type": "multihop",
        })
        print(f"[ok]  {page_key}: {qa['q']}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(questions)} multi-hop questions to {OUT_FILE}")
    print("Over-generated; manually keep the 20-30 that genuinely require two facts.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="pages to sample (over-generate)")
    args = ap.parse_args()
    main(args.n)