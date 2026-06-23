"""
Three-way figure comparison: baseline vs caption-mediated multimodal vs VLM.

For figure questions, the answer lives in a figure or table. Three systems:
  baseline   - text-only RAG; cannot retrieve images, so retrieval is ~0 by design
  mm_caption - CLIP retrieves the figure, and the figure's CAPTION is fed to the
               generator (caption-mediated multimodal)
  vlm        - CLIP retrieves the figure, and the ACTUAL IMAGE is passed to gpt-4o
               vision alongside the retrieved text (true visual reasoning)

mm_caption and vlm use the same CLIP retrieval, so their retrieval metrics match;
the difference in answer accuracy isolates "reading the caption" vs "reading the
image pixels".

Metrics: Recall@k and MRR (retrieval); answer accuracy, faithfulness, relevancy
(generation, judged by gpt-4o).

Usage:
    python compare_multimodal.py questions_publaynet_figures.json

Writes two timestamped CSVs into results/ (nothing overwritten).
"""
import sys
import json
import os
import csv
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from rag_basics import build_index, retrieve, generate
from rag_multimodal import build_image_index, retrieve_images, ask_multimodal
from rag_vlm import ask_vlm

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGE_MODEL = "gpt-4o"      # judge differs from the gpt-4o-mini generator, to avoid self-grading
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


def score_system(agg, name, question, expected_ans, answer, ranked, context, expected_src):
    """Judge one system's answer, add to its running totals, and return the scores."""
    rec, mrr = retrieval_metrics(ranked, expected_src)
    acc   = judge_answer(question, expected_ans, answer)
    faith = judge_faithfulness(answer, context)
    rel   = judge_relevancy(question, answer)
    agg[name]["recall"] += rec
    agg[name]["mrr"]    += mrr
    agg[name]["acc"]    += acc
    agg[name]["faith"]  += faith
    agg[name]["rel"]    += rel
    return acc, faith, rel, rec


SYSTEMS = ("baseline", "mm_caption", "vlm")
METRICS = ("recall", "mrr", "acc", "faith", "rel")


def main(qfile):
    with open(qfile, encoding="utf-8") as f:
        questions = json.load(f)
    qset = os.path.splitext(os.path.basename(qfile))[0]
    run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # colon-free for Windows filenames

    text_index, chunks, sources = build_index()
    img_index, img_names, captions = build_image_index()

    agg = {s: {m: 0 for m in METRICS} for s in SYSTEMS}
    detail_rows = []

    print(f"\nThree-way figure comparison on {qfile} ({len(questions)} questions)\n" + "=" * 72)
    for item in questions:
        question, expected_src, expected_ans = item["q"], item["source"], item["answer"]

        # baseline: text-only, scored over text retrieval (so ~0 on figure questions)
        base_ret = retrieve(question, text_index, chunks, sources, k=K)
        base_ans = generate(question, base_ret)
        base_ranked = [s for _, s, _ in base_ret]
        base_ctx = "\n\n".join(chunk for _, _, chunk in base_ret)
        base_acc, base_faith, base_rel, _ = score_system(
            agg, "baseline", question, expected_ans, base_ans, base_ranked, base_ctx, expected_src)

        # caption-mediated: CLIP retrieves the figure, its caption goes to the generator
        cap_ans, cap_text_ret, cap_img_ret = ask_multimodal(
            question, text_index, chunks, sources, img_index, img_names, captions, k=K)
        img_ranked = [name for _, name, _ in cap_img_ret]
        cap_ctx = "\n\n".join(chunk for _, _, chunk in cap_text_ret) + "\n" + \
                  "\n".join(cap for _, _, cap in cap_img_ret)
        cap_acc, cap_faith, cap_rel, cap_rec = score_system(
            agg, "mm_caption", question, expected_ans, cap_ans, img_ranked, cap_ctx, expected_src)

        # VLM: same CLIP retrieval, but the actual image is passed to gpt-4o vision
        vlm_ans, vlm_text_ret, vlm_img, vlm_score, vlm_cap = ask_vlm(
            question, text_index, chunks, sources, img_index, img_names, captions)
        vlm_ranked = [vlm_img] if vlm_img else []
        vlm_ctx = "\n\n".join(chunk for _, _, chunk in vlm_text_ret) + \
                  ("\n[IMAGE: " + vlm_img + "]" if vlm_img else "")
        vlm_acc, vlm_faith, vlm_rel, vlm_rec = score_system(
            agg, "vlm", question, expected_ans, vlm_ans, vlm_ranked, vlm_ctx, expected_src)

        figure_found = "✓" if cap_rec else "✗"
        top_figure = img_ranked[0] if img_ranked else "-"
        print(f"  base={base_acc} cap={cap_acc} vlm={vlm_acc} "
              f"fig{figure_found}={top_figure[:22]:22s} {question[:24]}")

        detail_rows.append({
            "question": question, "expected_source": expected_src,
            "retrieved_figure": top_figure, "figure_correct": cap_rec,
            "baseline_acc": base_acc, "baseline_faith": base_faith, "baseline_rel": base_rel,
            "mm_caption_acc": cap_acc, "mm_caption_faith": cap_faith, "mm_caption_rel": cap_rel,
            "vlm_acc": vlm_acc, "vlm_faith": vlm_faith, "vlm_rel": vlm_rel,
            "mm_caption_answer": cap_ans.replace("\n", " ")[:160],
            "vlm_answer": vlm_ans.replace("\n", " ")[:160],
        })

    n = len(questions)
    print("\n" + "=" * 72)
    print(f"{'metric':<16}{'baseline':>14}{'mm_caption':>14}{'vlm':>14}")
    print("-" * 58)
    for m, label in [("recall", f"Recall@{K}"), ("mrr", "MRR"),
                     ("acc", "AnswerAcc"), ("faith", "Faithfulness"), ("rel", "AnswerRel")]:
        print(f"{label:<16}" + "".join(f"{agg[s][m]/n:>14.3f}" for s in SYSTEMS))
    print("\nNote: baseline retrieval ~0 on figure questions by design. mm_caption and "
          "vlm share the same CLIP-retrieved figure, so their retrieval metrics match; "
          "the answer-accuracy gap isolates caption vs pixels.")

    # aggregate summary (3 rows)
    summary_csv = os.path.join(RESULTS_DIR, f"summary_multimodal_{qset}_{stamp}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "question_set", "n_questions", "system",
                                          "recall", "mrr", "acc", "faith", "rel"])
        w.writeheader()
        for s in SYSTEMS:
            w.writerow({
                "run_id": run_id, "question_set": qset, "n_questions": n, "system": s,
                "recall": round(agg[s]["recall"]/n, 3),
                "mrr":    round(agg[s]["mrr"]/n, 3),
                "acc":    round(agg[s]["acc"]/n, 3),
                "faith":  round(agg[s]["faith"]/n, 3),
                "rel":    round(agg[s]["rel"]/n, 3),
            })

    detail_csv = os.path.join(RESULTS_DIR, f"multimodal_{qset}_detail_{stamp}.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    print(f"\nSummary written to {summary_csv}")
    print(f"Per-question detail written to {detail_csv}")


if __name__ == "__main__":
    qfile = sys.argv[1] if len(sys.argv) > 1 else "questions_publaynet_figures.json"
    main(qfile)