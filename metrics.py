"""
Consolidated metric study: all nine metrics on the same answers, side by side,
with the rationale for which we adopt as primary and which we exclude.

This single script computes, on the BASELINE text RAG over a question set:

  PRIMARY SUITE (adopted) -------------------------------------------------
  Retrieval (objective, from the ranked source list):
    * Recall@k   : expected source in top-k?
    * MRR        : 1/rank of the correct source
    * NDCG@k     : ranking quality, graded position discount
  Generation / RAG (LLM-judged, gpt-4o):
    * Answer accuracy : factually correct vs expected?
    * Faithfulness    : supported by retrieved context (no hallucination)?
    * Answer relevancy: addresses the question?

  COMPARABILITY METRICS (computed, then EXCLUDED) -------------------------
    * BLEU         : n-gram overlap (translation metric)
    * ROUGE-L      : longest-common-subsequence overlap
    * BERTScore F1 : contextual-embedding similarity

Setup:
    pip install sacrebleu rouge-score bert-score
Usage:
    python metrics.py questions_publaynet_text.json
Writes a timestamped detail CSV and summary CSV into results/.
"""
import sys
import json
import os
import csv
import math
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score

from rag_basics import build_index, retrieve, generate

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
JUDGE_MODEL = "gpt-4o"
K = 3
LOW_THRESHOLD = 0.3   # below this, a surface metric counts as scoring an answer "low"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- LLM judges (primary generation metrics) ----------
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


# ---------- retrieval metrics (primary) ----------
def retrieval_metrics(ranked, expected):
    # (recall@k, MRR, NDCG@k) from where the expected source sits in the ranked list
    if expected in ranked:
        rank = ranked.index(expected) + 1
        return 1, 1.0 / rank, 1.0 / math.log2(rank + 1)
    return 0, 0.0, 0.0


def main(qfile):
    with open(qfile, encoding="utf-8") as f:
        questions = json.load(f)
    qset = os.path.splitext(os.path.basename(qfile))[0]
    run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # filename-safe (no colons)

    text_index, chunks, sources = build_index()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Pass 1: retrieve and generate every answer first, so BERTScore can run batched
    # in a single call (it loads a model and is far faster batched than per-item).
    refs, cands, records = [], [], []
    print("Generating answers + retrieval metrics...")
    for item in questions:
        question, expected_src, expected = item["q"], item["source"], item["answer"]
        retrieved = retrieve(question, text_index, chunks, sources, k=K)
        ranked = [source for _, source, _ in retrieved]
        context = "\n\n".join(chunk for _, _, chunk in retrieved)
        answer = generate(question, retrieved)
        recall, mrr, ndcg = retrieval_metrics(ranked, expected_src)
        refs.append(expected); cands.append(answer)
        records.append({"q": question, "exp": expected, "ans": answer, "ctx": context,
                        "recall": recall, "mrr": mrr, "ndcg": ndcg})

    # BERTScore over all answers at once
    print("Computing BERTScore (downloads a model on first run)...")
    _, _, bert_f1 = bertscore_score(cands, refs, lang="en", verbose=False)

    # Pass 2: per-item LLM judges plus the surface-overlap metrics
    print("Computing BLEU, ROUGE-L, and LLM judges per item...\n")
    totals = {k: 0.0 for k in ("recall","mrr","ndcg","acc","faith","rel","bleu","rougeL","bert")}
    rows, divergences = [], []

    for i, record in enumerate(records):
        question, expected, answer, context = record["q"], record["exp"], record["ans"], record["ctx"]

        # primary generation metrics
        acc   = judge_answer(question, expected, answer)
        faith = judge_faithfulness(answer, context)
        rel   = judge_relevancy(question, answer)
        # comparability metrics (computed for the study, excluded from the headline suite)
        bleu   = sacrebleu.sentence_bleu(answer, [expected]).score / 100.0
        rougeL = rouge.score(expected, answer)["rougeL"].fmeasure
        bert   = float(bert_f1[i])

        for key, value in [("recall", record["recall"]), ("mrr", record["mrr"]), ("ndcg", record["ndcg"]),
                           ("acc", acc), ("faith", faith), ("rel", rel),
                           ("bleu", bleu), ("rougeL", rougeL), ("bert", bert)]:
            totals[key] += value

        rows.append({
            "question": question, "expected": expected, "generated": answer,
            "recall": record["recall"], "mrr": round(record["mrr"],3), "ndcg": round(record["ndcg"],3),
            "answer_correct": acc, "faithfulness": faith, "answer_relevancy": rel,
            "bleu": round(bleu,3), "rougeL": round(rougeL,3), "bertscore_f1": round(bert,3),
        })

        # a correct answer that both surface metrics score low is the divergence we want to surface
        if acc == 1 and bleu < LOW_THRESHOLD and rougeL < LOW_THRESHOLD:
            divergences.append((question, expected, answer, bleu, rougeL, bert))

        print(f"  acc={acc} faith={faith} rel={rel} | bleu={bleu:.2f} "
              f"rougeL={rougeL:.2f} bert={bert:.2f}  {question[:34]}")

    n = len(questions)
    print("\n==== ALL NINE METRICS (baseline, n={}) ====".format(n))
    print("--- PRIMARY: retrieval ---")
    print(f"Recall@{K}         : {totals['recall']/n:.3f}")
    print(f"MRR              : {totals['mrr']/n:.3f}")
    print(f"NDCG@{K}           : {totals['ndcg']/n:.3f}")
    print("--- PRIMARY: generation / RAG ---")
    print(f"Answer accuracy  : {totals['acc']/n:.3f}")
    print(f"Faithfulness     : {totals['faith']/n:.3f}")
    print(f"Answer relevancy : {totals['rel']/n:.3f}")
    print("--- COMPARABILITY (computed, then excluded) ---")
    print(f"BLEU             : {totals['bleu']/n:.3f}")
    print(f"ROUGE-L          : {totals['rougeL']/n:.3f}")
    print(f"BERTScore F1     : {totals['bert']/n:.3f}")

    print(f"\n==== DIVERGENCE: judge=correct but BLEU & ROUGE low ({len(divergences)}) ====")
    print("Correct answers penalised by surface-overlap metrics (valid paraphrases):\n")
    for question, expected, answer, bleu, rougeL, bert in divergences[:8]:
        print(f"Q: {question[:68]}")
        print(f"  expected : {expected[:68]}")
        print(f"  generated: {answer[:68]}")
        print(f"  bleu={bleu:.2f} rougeL={rougeL:.2f} bert={bert:.2f} judge=1\n")

    # ----- timestamped per-item detail (never overwritten) -----
    detail_csv = os.path.join(RESULTS_DIR, f"metric_comparison_{qset}_detail_{stamp}.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ----- timestamped summary of all nine aggregate metrics, tagged by tier -----
    summary_csv = os.path.join(RESULTS_DIR, f"metric_comparison_{qset}_summary_{stamp}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "question_set", "n_questions",
                                               "metric", "tier", "value"])
        writer.writeheader()
        tiers = {
            "recall": "primary_retrieval", "mrr": "primary_retrieval", "ndcg": "primary_retrieval",
            "acc": "primary_generation", "faith": "primary_generation", "rel": "primary_generation",
            "bleu": "comparability_excluded", "rougeL": "comparability_excluded", "bert": "comparability_excluded",
        }
        labels = {"recall": f"Recall@{K}", "mrr": "MRR", "ndcg": f"NDCG@{K}",
                  "acc": "AnswerAccuracy", "faith": "Faithfulness", "rel": "AnswerRelevancy",
                  "bleu": "BLEU", "rougeL": "ROUGE-L", "bert": "BERTScore_F1"}
        for key in ("recall","mrr","ndcg","acc","faith","rel","bleu","rougeL","bert"):
            writer.writerow({"run_id": run_id, "question_set": qset, "n_questions": n,
                             "metric": labels[key], "tier": tiers[key],
                             "value": round(totals[key]/n, 3)})

    print(f"Per-item detail   -> {detail_csv}")
    print(f"Nine-metric summary -> {summary_csv}")
    print("\nRATIONALE: high LLM-judged accuracy alongside low BLEU/ROUGE on the SAME "
          "answers shows surface-overlap metrics penalise valid paraphrases. BERTScore "
          "(embedding-based) is better but still imperfect. We therefore adopt the six "
          "primary metrics and exclude BLEU/ROUGE (and Exact Match) for open-form QA.")


if __name__ == "__main__":
    qfile = sys.argv[1] if len(sys.argv) > 1 else "questions_publaynet_text.json"
    main(qfile)