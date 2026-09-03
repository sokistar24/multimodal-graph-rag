"""
Single-system evaluation: runs the text-only baseline RAG on a question set and
reports the five metrics. Useful for a quick check of one system; the four-way
comparison (compare_all.py) is the main experiment.

Retrieval (from the ranked source list):
  Recall@k - was the expected source among the top-k chunks?
  MRR      - 1/rank of the first correct source (rewards ranking it high)
Generation (judged by gpt-4o, a different model from the gpt-4o-mini generator):
  Answer accuracy - factually correct vs the expected answer?
  Faithfulness    - supported by the retrieved context (no hallucination)?
  Answer relevancy- does it address the question?

Usage:
    python run_eval.py                                  # defaults to the text questions
    python run_eval.py questions_publaynet_text.json
Writes a timestamped CSV into results/.
"""
import os
import sys
import json
import csv
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from rag_basics import build_index, retrieve, generate

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGE_MODEL = "gpt-4.1"      # judge differs from the gpt-4o generator, to avoid self-grading
K = 3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def retrieval_metrics(retrieved, expected_source):
    """retrieved is a list of (score, source, chunk), best first.
    Returns hit (Recall@k, 0/1), rank, and reciprocal rank."""
    sources = [src for _, src, _ in retrieved]
    if expected_source in sources:
        rank = sources.index(expected_source) + 1
        return True, rank, 1.0 / rank
    return False, None, 0.0


# ---------- LLM judges (all return 1/0) ----------
def _judge(prompt):
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return 1 if resp.choices[0].message.content.strip().startswith("1") else 0

def judge_answer(question, expected, generated):
    """Correctness: is the answer factually right vs the expected answer?"""
    return _judge(
        "You are grading a question-answering system. Decide whether the GENERATED "
        "answer is factually correct, given the EXPECTED answer. Minor wording "
        "differences are fine. Reply 1 if correct, 0 if not.\n\n"
        f"QUESTION: {question}\nEXPECTED: {expected}\nGENERATED: {generated}\n\nGrade (1 or 0):")

def judge_faithfulness(generated, context):
    """Groundedness: is the answer supported by the retrieved context?"""
    return _judge(
        "Check whether an answer is FAITHFUL to the context (every claim supported, "
        "nothing invented). Reply 1 if fully supported, 0 otherwise.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{generated}\n\nGrade (1 or 0):")

def judge_relevancy(question, generated):
    """Relevancy: does the answer address the question, regardless of correctness?"""
    return _judge(
        "Check whether an answer is RELEVANT to the question (addresses it, "
        "regardless of correctness). Reply 1 if relevant, 0 otherwise.\n\n"
        f"QUESTION: {question}\nANSWER: {generated}\n\nGrade (1 or 0):")


def main(qfile):
    with open(qfile, encoding="utf-8") as f:
        questions = json.load(f)
    qset = os.path.splitext(os.path.basename(qfile))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # colon-free for Windows filenames

    text_index, chunks, sources = build_index()

    rows = []
    totals = {"recall": 0, "mrr": 0, "acc": 0, "faith": 0, "rel": 0}

    for item in questions:
        question, expected_src, expected_ans = item["q"], item["source"], item["answer"]

        retrieved = retrieve(question, text_index, chunks, sources, k=K)
        hit, rank, rr = retrieval_metrics(retrieved, expected_src)

        answer = generate(question, retrieved)
        context = "\n\n".join(chunk for _, _, chunk in retrieved)

        acc = judge_answer(question, expected_ans, answer)
        faith = judge_faithfulness(answer, context)
        rel = judge_relevancy(question, answer)

        totals["recall"] += int(hit)
        totals["mrr"]    += rr
        totals["acc"]    += acc
        totals["faith"]  += faith
        totals["rel"]    += rel

        rows.append({
            "question": question,
            "expected_source": expected_src,
            "retrieved_sources": " > ".join(s for _, s, _ in retrieved),
            "hit": int(hit),
            "rank": rank if rank else "",
            "reciprocal_rank": round(rr, 3),
            "answer_correct": acc,
            "faithfulness": faith,
            "answer_relevancy": rel,
            "generated_answer": answer.replace("\n", " ")[:200],
        })
        flag = "ok " if hit else "MISS"
        print(f"[{flag}] rr={rr:.2f} acc={acc} faith={faith} rel={rel}  {question[:42]}")

    n = len(questions)
    print("\n==== SUMMARY ====")
    print(f"Questions        : {n}")
    print(f"Recall@{K}        : {totals['recall']/n:.3f}")
    print(f"MRR              : {totals['mrr']/n:.3f}")
    print(f"Answer accuracy  : {totals['acc']/n:.3f}")
    print(f"Faithfulness     : {totals['faith']/n:.3f}")
    print(f"Answer relevancy : {totals['rel']/n:.3f}")

    out_csv = os.path.join(RESULTS_DIR, f"eval_{qset}_{stamp}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-question detail written to {out_csv}")


if __name__ == "__main__":
    qfile = sys.argv[1] if len(sys.argv) > 1 else "questions_publaynet_text.json"
    main(qfile)