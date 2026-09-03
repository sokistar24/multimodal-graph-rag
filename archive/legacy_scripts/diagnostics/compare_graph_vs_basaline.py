"""
Baseline (text-only RAG) vs +KG (graph-augmented RAG).

Runs both systems on one question set with the same judges. Most useful on the
multi-hop questions, where the KG can help by supplying facts that connect the two
pieces of evidence a question needs.

Usage:
    python compare_graph_vs_baseline.py questions_publaynet_multihop.json
    python compare_graph_vs_baseline.py questions_publaynet_text.json   # control: should tie

Metrics: Recall@k and MRR (retrieval); answer accuracy, faithfulness, relevancy
(generation, judged by gpt-4.1). Both systems share the same retriever, so their
retrieval metrics are identical; the KG only changes what reaches the generator.

Writes two timestamped CSVs into results/ (nothing overwritten):
  summary_graph_vs_baseline_<qset>_<timestamp>.csv  - aggregate, 2 rows
  graph_vs_baseline_<qset>_detail_<timestamp>.csv   - per-question scores
"""
import sys
import json
import os
import csv
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from rag_basics import build_index, retrieve, generate
from graph_aware import build_triples, build_graph, graph_facts_for_query, generate_with_graph

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGE_MODEL = "gpt-4.1"      # judge differs from the gpt-4o generator, to avoid self-grading
K = 3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- LLM judges (all return 1/0) ----------
def _judge(prompt):
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return 1 if resp.choices[0].message.content.strip().startswith("1") else 0

def judge_answer(question, expected, generated):
    return _judge(
        "You are grading a question-answering system. Decide whether the GENERATED "
        "answer is factually correct, given the EXPECTED answer. Minor wording "
        "differences are fine. Reply 1 if correct, 0 if not.\n\n"
        f"QUESTION: {question}\nEXPECTED: {expected}\nGENERATED: {generated}\n\nGrade (1 or 0):")

def judge_faithfulness(generated, context):
    return _judge(
        "Check whether an answer is FAITHFUL to the context (every claim supported, "
        "nothing invented). Reply 1 if fully supported, 0 otherwise.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{generated}\n\nGrade (1 or 0):")

def judge_relevancy(question, generated):
    return _judge(
        "Check whether an answer is RELEVANT to the question (addresses it, "
        "regardless of correctness). Reply 1 if relevant, 0 otherwise.\n\n"
        f"QUESTION: {question}\nANSWER: {generated}\n\nGrade (1 or 0):")


def retrieval_metrics(ranked_sources, expected_source):
    """Recall@k and MRR for a single relevant source. ranked_sources is best-first."""
    if expected_source in ranked_sources:
        rank = ranked_sources.index(expected_source) + 1
        return 1, 1.0 / rank
    return 0, 0.0


METRICS = ("recall", "mrr", "acc", "faith", "rel")


def main(qfile):
    with open(qfile, encoding="utf-8") as f:
        questions = json.load(f)
    qset = os.path.splitext(os.path.basename(qfile))[0]
    run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # colon-free for Windows filenames

    text_index, chunks, sources = build_index()
    graph = build_graph(build_triples())

    agg = {"baseline": {m: 0 for m in METRICS},
           "graph":    {m: 0 for m in METRICS}}
    detail_rows = []

    print(f"\nBaseline vs +KG on {qfile} ({len(questions)} questions)\n" + "=" * 64)
    for item in questions:
        question, expected_src, expected_ans = item["q"], item["source"], item["answer"]

        # both systems use the same retrieved text; only the +KG one adds graph facts
        retrieved = retrieve(question, text_index, chunks, sources, k=K)
        ranked = [s for _, s, _ in retrieved]
        context = "\n\n".join(chunk for _, _, chunk in retrieved)
        rec, mrr = retrieval_metrics(ranked, expected_src)

        baseline_ans = generate(question, retrieved)
        baseline_acc = judge_answer(question, expected_ans, baseline_ans)

        facts = graph_facts_for_query(question, graph)
        graph_ans = generate_with_graph(question, retrieved, facts)
        graph_acc = judge_answer(question, expected_ans, graph_ans)

        row = {"question": question, "expected_source": expected_src, "facts_injected": len(facts)}
        for name, answer, acc in [("baseline", baseline_ans, baseline_acc),
                                  ("graph", graph_ans, graph_acc)]:
            faith = judge_faithfulness(answer, context + "\n" + "\n".join(facts))
            rel = judge_relevancy(question, answer)
            agg[name]["recall"] += rec
            agg[name]["mrr"]    += mrr
            agg[name]["acc"]    += acc
            agg[name]["faith"]  += faith
            agg[name]["rel"]    += rel
            row[f"{name}_acc"] = acc
            row[f"{name}_faith"] = faith
            row[f"{name}_rel"] = rel
            row[f"{name}_answer"] = answer.replace("\n", " ")[:160]
        detail_rows.append(row)

        # quick per-question marker: did the KG help, hurt, or tie on accuracy?
        mark = "KG+" if graph_acc > baseline_acc else ("KG-" if graph_acc < baseline_acc else "=")
        print(f"  {mark:3s} base={baseline_acc} kg={graph_acc} facts={len(facts)}  {question[:42]}")

    n = len(questions)
    print("\n" + "=" * 64)
    print(f"{'metric':<16}{'baseline':>12}{'+KG':>12}")
    print("-" * 40)
    for m, label in [("recall", f"Recall@{K}"), ("mrr", "MRR"),
                     ("acc", "AnswerAcc"), ("faith", "Faithfulness"), ("rel", "AnswerRel")]:
        print(f"{label:<16}{agg['baseline'][m]/n:>12.3f}{agg['graph'][m]/n:>12.3f}")

    # aggregate summary (2 rows)
    summary_csv = os.path.join(RESULTS_DIR, f"summary_graph_vs_baseline_{qset}_{stamp}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "question_set", "n_questions", "system",
                                          "recall", "mrr", "acc", "faith", "rel"])
        w.writeheader()
        for name, label in [("baseline", "baseline"), ("graph", "+KG")]:
            w.writerow({
                "run_id": run_id, "question_set": qset, "n_questions": n, "system": label,
                "recall": round(agg[name]["recall"]/n, 3),
                "mrr":    round(agg[name]["mrr"]/n, 3),
                "acc":    round(agg[name]["acc"]/n, 3),
                "faith":  round(agg[name]["faith"]/n, 3),
                "rel":    round(agg[name]["rel"]/n, 3),
            })

    # per-question detail (includes how many graph facts were injected each time)
    detail_csv = os.path.join(RESULTS_DIR, f"graph_vs_baseline_{qset}_detail_{stamp}.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    print(f"\nSummary written to {summary_csv}")
    print(f"Per-question detail written to {detail_csv}")


if __name__ == "__main__":
    qfile = sys.argv[1] if len(sys.argv) > 1 else "questions_publaynet_multihop.json"
    main(qfile)