"""
Generates text-grounded questions from the ingested PubLayNet corpus.

A subset of pages is sampled, and for each the most information-rich chunk is given
to the model, which writes one question answerable only from that chunk plus its
answer. The source is known by construction (the page the chunk came from), so the
questions work as ground truth. The set is over-generated and then skimmed down.

Questions are tagged "type": "text" so they sit alongside the figure and multi-hop
sets in one consistently-structured collection for the four-way ablation. The output
format matches the other question files, so run_eval.py and compare_all.py read it
unchanged.

Usage:
    python generate_questions_publaynet.py --n 45
"""
import os
import glob
import json
import random
import argparse
from dotenv import load_dotenv
from openai import OpenAI

from rag_basics import chunk_text   # reuse the same chunker the pipeline uses

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

GEN_MODEL = "gpt-4o-mini"
CORPUS_DIR = "publaynet_corpus"
OUT_FILE = "questions_publaynet_text.json"


def pick_chunk(text):
    """Returns the longest chunk of a page (the most information-rich, so it gives
    the model the best material to form a specific question)."""
    chunks = chunk_text(text)
    return max(chunks, key=len) if chunks else text


# Few-shot examples: real high-quality questions from this corpus that model the
# target style -- a specific question with a concrete, scientifically meaningful answer.
FEWSHOT = """Here are examples of GOOD questions and answers:

Example 1:
{"q": "What is the main mechanism by which fibrates exert their effects?", "answer": "activating the peroxisome proliferator-activated receptor-alpha (PPAR-alpha)"}

Example 2:
{"q": "What is of paramount importance for the activation of p70S6K?", "answer": "the phosphorylation of Serine residue in 411 position"}

Example 3:
{"q": "What factors are associated with the stem cell-like phenotype in NSCLC resistance to chemotherapy?", "answer": "Sox2, Oct and Nanog"}

Notice these questions are SPECIFIC, ask about a concrete scientific fact (a mechanism,
a precise detail, or named entities), and have answers explicitly stated in the source.
"""


def generate_qa(chunk):
    """Returns one {"q", "answer"} answerable from the chunk, or None if the text is
    too garbled to yield a clear scientific fact (model replies SKIP)."""
    prompt = (
        FEWSHOT +
        "\nNow read the TEXT below (extracted via OCR from a scientific paper, so it may "
        "contain minor errors) and write ONE question in the same style as the examples, "
        "answerable using ONLY this text, plus its short answer.\n"
        "Rules:\n"
        "- Match the specificity and scientific concreteness of the examples.\n"
        "- The answer MUST be explicitly stated in the text.\n"
        "- Do NOT rely on outside knowledge, and do NOT ask vague or trivial questions.\n"
        "- If the text is too garbled or lacks a clear scientific fact, respond with "
        '{"q": "SKIP", "answer": "SKIP"}.\n'
        "- Respond with ONLY a JSON object: {\"q\": \"...\", \"answer\": \"...\"}\n\n"
        f"TEXT:\n{chunk}"
    )
    resp = client.chat.completions.create(
        model=GEN_MODEL, temperature=0.3,
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

    random.seed(42)                                  # fixed seed for reproducible sampling
    sampled = random.sample(files, min(n, len(files)))

    questions = []
    for path in sampled:
        page_key = os.path.splitext(os.path.basename(path))[0]   # e.g. PMC4991227_00003
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunk = pick_chunk(text)
        qa = generate_qa(chunk)
        if qa is None:
            print(f"[skip] {page_key}: text too garbled or generation failed")
            continue
        questions.append({
            "q": qa["q"],
            "source": f"{page_key}.txt",             # matches how build_index tags sources
            "answer": qa["answer"],
            "type": "text",
        })
        print(f"[ok]  {page_key}: {qa['q']}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(questions)} questions to {OUT_FILE}")
    print("Over-generated; skim and delete any weak or garbled questions before evaluating.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45,
                    help="pages to sample (over-generate, then skim down to ~25)")
    args = ap.parse_args()
    main(args.n)