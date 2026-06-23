"""
Four-way ablation: the main comparison in this project.

Runs all four systems on one question set, scoring each with the same judges:
  baseline    - text-only RAG
  +KG         - text + knowledge-graph facts
  +multimodal - text + CLIP-retrieved figure captions
  +both       - text + KG + figure (the full enhanced system)

Run it once per question set:
    python compare_all.py questions_publaynet_text.json
    python compare_all.py questions_publaynet_figures.json
    python compare_all.py questions_publaynet_multihop.json

Each run writes two timestamped CSVs into results/ (nothing is overwritten, so
runs from before and after a change can be compared):
  summary_<qset>_<timestamp>.csv            - aggregate metrics, one row per system
  compare_all_<qset>_detail_<timestamp>.csv - per-question scores and answers

Metrics: Recall@k and MRR (retrieval); answer accuracy, faithfulness, relevancy
(generation, judged by gpt-4.1). Retrieval is modality-aware: figure questions
(.png source) are scored over the image ranking, text questions over the text ranking.
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
from rag_multimodal import build_image_index, retrieve_images, generate_multimodal
from rag_full import generate_full

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGE_MODEL = "gpt-4.1"      # judge differs from the gpt-4o-mini generator, to avoid self-grading
K = 3                      # chunks/images retrieved per question
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


# ---------- the four systems ----------
# Each takes the shared indices/graph and returns:
#   (answer, text_sources_ranked, image_names_ranked_or_None, context_used)
# text-only systems return None for the image ranking.
def run_baseline(question, text_index, chunks, sources, graph, img_index, img_names, captions):
    retrieved = retrieve(question, text_index, chunks, sources, k=K)
    context = "\n\n".join(chunk for _, _, chunk in retrieved)
    return generate(question, retrieved), [s for _, s, _ in retrieved], None, context

def run_kg(question, text_index, chunks, sources, graph, img_index, img_names, captions):
    retrieved = retrieve(question, text_index, chunks, sources, k=K)
    facts = graph_facts_for_query(question, graph)
    context = "\n\n".join(chunk for _, _, chunk in retrieved) + "\n" + "\n".join(facts)
    return generate_with_graph(question, retrieved, facts), [s for _, s, _ in retrieved], None, context

def run_multimodal(question, text_index, chunks, sources, graph, img_index, img_names, captions):
    retrieved = retrieve(question, text_index, chunks, sources, k=K)
    images = retrieve_images(question, img_index, img_names, captions, k=K)
    context = "\n\n".join(chunk for _, _, chunk in retrieved) + "\n" + \
              "\n".join(cap for _, _, cap in images)
    return (generate_multimodal(question, retrieved, images),
            [s for _, s, _ in retrieved], [name for _, name, _ in images], context)

def run_both(question, text_index, chunks, sources, graph, img_index, img_names, captions):
    retrieved = retrieve(question, text_index, chunks, sources, k=K)
    facts = graph_facts_for_query(question, graph)
    images = retrieve_images(question, img_index, img_names, captions, k=K)
    context = ("\n\n".join(chunk for _, _, chunk in retrieved) + "\n" + "\n".join(facts) +
               "\n" + "\n".join(cap for _, _, cap in images))
    return (generate_full(question, retrieved, facts, images),
            [s for _, s, _ in retrieved], [name for _, name, _ in images], context)

SYSTEMS = [
    ("baseline",    run_baseline),
    ("+KG",         run_kg),
    ("+multimodal", run_multimodal),
    ("+both",       run_both),
]
METRICS = ("recall", "mrr", "acc", "faith", "rel")


def main(qfile):
    with open(qfile, encoding="utf-8") as f:
        questions = json.load(f)
    qset = os.path.splitext(os.path.basename(qfile))[0]
    run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # colon-free for Windows filenames

    # Build everything once, shared across all four systems.
    text_index, chunks, sources = build_index()
    graph = build_graph(build_triples())
    img_index, img_names, captions = build_image_index()

    agg = {name: {m: 0 for m in METRICS} for name, _ in SYSTEMS}
    detail_rows = []

    print(f"\nFour-way ablation on {qfile} ({len(questions)} questions)\n" + "=" * 72)
    for item in questions:
        question, expected_src, expected_ans = item["q"], item["source"], item["answer"]
        is_image_question = expected_src.endswith(".png")   # figure questions have a .png source
        row = {"question": question, "expected_source": expected_src}

        for name, run_system in SYSTEMS:
            answer, text_ranked, img_ranked, context = run_system(
                question, text_index, chunks, sources, graph, img_index, img_names, captions)

            # score retrieval against whichever modality the question's answer lives in
            if is_image_question:
                ranked = img_ranked if img_ranked is not None else []
            else:
                ranked = text_ranked
            rec, mrr = retrieval_metrics(ranked, expected_src)

            acc = judge_answer(question, expected_ans, answer)
            faith = judge_faithfulness(answer, context)
            rel = judge_relevancy(question, answer)

            agg[name]["recall"] += rec
            agg[name]["mrr"]    += mrr
            agg[name]["acc"]    += acc
            agg[name]["faith"]  += faith
            agg[name]["rel"]    += rel

            row[f"{name}_acc"] = acc
            row[f"{name}_faith"] = faith
            row[f"{name}_rel"] = rel
            row[f"{name}_rec"] = rec
            row[f"{name}_answer"] = answer.replace("\n", " ")[:160]

        detail_rows.append(row)
        print(f"  done: {question[:60]}")

    n = len(questions)

    # print the summary table to the console
    print("\n" + "=" * 72)
    print(f"{'metric':<14}" + "".join(f"{name:>14}" for name, _ in SYSTEMS))
    print("-" * 72)
    for m, label in [("recall", f"Recall@{K}"), ("mrr", "MRR"),
                     ("acc", "AnswerAcc"), ("faith", "Faithfulness"), ("rel", "AnswerRel")]:
        print(f"{label:<14}" + "".join(f"{agg[name][m]/n:>14.3f}" for name, _ in SYSTEMS))

    # save the aggregate summary (one row per system) and the per-question detail
    summary_csv = os.path.join(RESULTS_DIR, f"summary_{qset}_{stamp}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "question_set", "n_questions", "system",
                                          "recall", "mrr", "acc", "faith", "rel"])
        w.writeheader()
        for name, _ in SYSTEMS:
            w.writerow({
                "run_id": run_id, "question_set": qset, "n_questions": n, "system": name,
                "recall": round(agg[name]["recall"]/n, 3),
                "mrr":    round(agg[name]["mrr"]/n, 3),
                "acc":    round(agg[name]["acc"]/n, 3),
                "faith":  round(agg[name]["faith"]/n, 3),
                "rel":    round(agg[name]["rel"]/n, 3),
            })

    detail_csv = os.path.join(RESULTS_DIR, f"compare_all_{qset}_detail_{stamp}.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    print(f"\nSummary written to {summary_csv}")
    print(f"Per-question detail written to {detail_csv}")
    print("\nNote: text-only systems (baseline, +KG) score ~0 retrieval on figure "
          "questions by design, since they cannot retrieve images.")


if __name__ == "__main__":
    qfile = sys.argv[1] if len(sys.argv) > 1 else "questions_publaynet_text.json"
    main(qfile)