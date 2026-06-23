"""
Evaluation harness for the RAG system.

RETRIEVAL (objective, computed from the ranked source list):
  * Recall@k : was the expected source among the top-k chunks?
  * MRR      : 1/rank of the first correct source (rewards ranking it high)

GENERATION / RAG (LLM-judged, gpt-4o -- a different model from the generator):
  * Answer accuracy : is the answer factually correct vs the expected answer?
  * Faithfulness    : is the answer supported by the retrieved context (no hallucination)?
  * Answer relevancy: does the answer actually address the question?

Setup:
    python run_eval.py
Reads:  questions_publaynet_text.json
Writes: eval_results.csv
"""
import os
import json
import csv
from dotenv import load_dotenv
from openai import OpenAI

from rag_basics import build_index, retrieve, generate

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
JUDGE_MODEL = "gpt-4o"
K = 3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- RETRIEVAL METRICS ----------
def retrieval_metrics(retrieved, expected_source):
    """retrieved is a list of (score, source, chunk), best first.
    Returns hit (Recall@k 0/1), rank, and reciprocal rank."""
    sources = [src for _, src, _ in retrieved]
    hit = expected_source in sources
    if hit:
        rank = sources.index(expected_source) + 1
        rr = 1.0 / rank
    else:
        rank, rr = None, 0.0
    return hit, rank, rr


# ---------- LLM JUDGES ----------
def _judge(prompt):
    """Single-character 1/0 judge call."""
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
        "differences are fine. Reply with exactly one character: 1 if correct, 0 if not.\n\n"
        f"QUESTION: {question}\nEXPECTED: {expected}\nGENERATED: {generated}\n\nGrade (1 or 0):"
    )


def judge_faithfulness(generated, context):
    """Groundedness: is the answer supported by the retrieved context (no hallucination)?"""
    return _judge(
        "You are checking whether an answer is FAITHFUL to the provided context, "
        "meaning every claim in the answer is supported by the context and nothing "
        "is invented. Reply with exactly one character: 1 if fully supported, 0 if it "
        "contains unsupported or hallucinated claims.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{generated}\n\nGrade (1 or 0):"
    )


def judge_relevancy(question, generated):
    """Relevancy: does the answer actually address the question asked?"""
    return _judge(
        "You are checking whether an answer is RELEVANT to the question, meaning it "
        "directly addresses what was asked (regardless of whether it is correct). "
        "An evasive or off-topic answer scores 0. Reply with exactly one character: "
        "1 if relevant, 0 if not.\n\n"
        f"QUESTION: {question}\nANSWER: {generated}\n\nGrade (1 or 0):"
    )


# ---------- RUN ----------
def main():
    with open("questions_publaynet_text.json", encoding="utf-8") as f:
        questions = json.load(f)

    index, chunks, sources = build_index()

    rows = []
    sums = {"recall": 0, "mrr": 0, "acc": 0, "faith": 0, "rel": 0}

    for item in questions:
        q, expected_src, expected_ans = item["q"], item["source"], item["answer"]

        retrieved = retrieve(q, index, chunks, sources, k=K)
        hit, rank, rr = retrieval_metrics(retrieved, expected_src)

        answer = generate(q, retrieved)
        context = "\n\n".join(chunk for _, _, chunk in retrieved)

        acc = judge_answer(q, expected_ans, answer)
        faith = judge_faithfulness(answer, context)
        rel = judge_relevancy(q, answer)

        sums["recall"] += int(hit)
        sums["mrr"] += rr
        sums["acc"] += acc
        sums["faith"] += faith
        sums["rel"] += rel

        rows.append({
            "question": q,
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
        print(f"[{flag}] rr={rr:.2f} acc={acc} faith={faith} rel={rel}  {q[:42]}")

    n = len(questions)
    print("\n==== SUMMARY ====")
    print(f"Questions          : {n}")
    print("--- Retrieval ---")
    print(f"Recall@{K}          : {sums['recall']/n:.3f}")
    print(f"MRR                : {sums['mrr']/n:.3f}")
    print("--- Generation / RAG ---")
    print(f"Answer accuracy    : {sums['acc']/n:.3f}")
    print(f"Faithfulness       : {sums['faith']/n:.3f}")
    print(f"Answer relevancy   : {sums['rel']/n:.3f}")

    with open(os.path.join(RESULTS_DIR, "eval_results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-question detail written to {os.path.join(RESULTS_DIR, 'eval_results.csv')}")


if __name__ == "__main__":
    main()