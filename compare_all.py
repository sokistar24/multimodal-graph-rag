"""
Four-way ablation, run once per generator. The main comparison in this project.

Runs all four systems on one question set with ONE generator, scoring each with
judges that sit outside the generator set:
  baseline    - text-only RAG
  +KG         - text + knowledge-graph facts
  +multimodal - text + CLIP-retrieved figure (pixels, or caption if text-only)
  +both       - text + KG + figure (the full enhanced system)

Usage — one run per (question set × generator):
    python compare_all.py questions_publaynet_text.json --model gpt4o-mini
    python compare_all.py questions_publaynet_figures.json --model llama4-scout
    python compare_all.py questions_publaynet_text.json --model gpt4o-mini --limit 10   # pilot

    # HotpotQA is image-free, so only the text-only pair applies:
    python compare_all.py questions_hotpotqa_bridge.json --model gpt4o-mini \
        --corpus hotpotqa_corpus --systems baseline,+KG

Generators (2 closed, 2 open — all production/cheap tier):
    gpt4o-mini         OpenAI     closed   vision
    gemini-flash-lite  Google     closed   vision  (thinking model — see note)
    llama4-maverick    DeepInfra  open     vision (FP8 serve)
    llama4-scout       Groq/Meta  open     vision

Judges — neither is a generator, so nothing grades its own family:
    deepseek      accuracy, relevancy, and faithfulness on TEXT evidence
    claude-haiku  faithfulness on FIGURE questions — DeepSeek is text-only and
                  cannot see the evidence it would be grading

Retrieval is built ONCE and shared by all four systems and all four generators,
so any difference in the results traces to the generator alone.

Outputs (timestamped, nothing overwritten):
  results/summary_<qset>_<model>_<stamp>.csv   - one row per system
  results/detail_<qset>_<model>_<stamp>.csv    - one row per question × system,
                                                 incl. tokens, latency, cost

Note on Gemini thinking tokens
──────────────────────────────
Gemini spends tokens on internal reasoning before visible output; they count as
output_tokens. We leave thinking ON — a practitioner pays for them, so it is the
decision-relevant number. Gemini's token/cost figures are therefore not strictly
like-for-like with non-thinking models; the paper footnotes this.
"""
import os
import csv
import sys
import json
import argparse
from datetime import datetime

from dotenv import load_dotenv

from llm_client import GENERATORS, DIAGNOSTICS, MODELS, judge
from rag_basics import (build_index, retrieve, generate,
                        CORPUS_DIR as DEFAULT_CORPUS_DIR)
from graph_aware import build_triples, build_graph, graph_facts_for_query, generate_with_graph
from rag_multimodal import build_image_index, retrieve_images, generate_multimodal
from rag_full import generate_full

load_dotenv()

K            = 3                  # text chunks retrieved per question

# ---------------------------------------------------------------------------
# Image knobs. These were ONE constant (K_IMAGES = 1), which conflated three
# separate decisions and made the metrics unreportable.
#
# The pilot's k=3 blowup (458 -> 26,450 input tokens, $0.008 -> $0.399 per 100q)
# was caused by SENDING three base64 crops, not by RETRIEVING three. Retrieval is
# free: CLIP already scores all ~600 images, so taking the top 5 costs nothing.
# Collapsing both into one constant meant capping the payload also capped the
# scoring depth — and at depth 1, Recall@k, AllGoldFound and MRR degenerate to
# the same number (you either got the one image or you didn't, and if you did it
# is rank 1 by definition). That is exactly what every run so far reported:
# three "different" metrics printing identical values to three decimals.
#
# Split, each knob now does one job:
K_IMAGES_RETRIEVE = 5   # scored for Recall@3/MRR/NDCG. Free — CLIP ranks all
                        # images anyway. Must be >= 3 for Recall@3 to mean
                        # Recall@3 rather than Recall@1 wearing the wrong label.
K_IMAGES_CAPTION  = 3   # captions injected on the caption-mediated path
                        # (~40 tok each, so ~120 tok). Only used when pixels are
                        # NOT sent — generate_multimodal/generate_full replace
                        # the caption block with the image, they don't stack.
K_IMAGES_PIXELS   = 1   # base64 crops sent to a vision model. THIS is the
                        # expensive knob (~1,000 tok each). Keep at 1.

RESULTS_DIR  = "results"
IMAGE_DIR    = "publaynet_images"
TEXT_JUDGE   = "deepseek"         # accuracy, relevancy, text faithfulness
VISION_JUDGE = "claude-haiku"     # faithfulness on figure questions

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- judges (all return 1/0) ----------
def judge_answer(question, expected, generated):
    """Graded against the human-reviewed reference answer — no evidence needed."""
    v, r = judge(TEXT_JUDGE,
        "You are grading a question-answering system. Decide whether the GENERATED "
        "answer is factually correct, given the EXPECTED answer. Minor wording "
        "differences are fine. Reply with ONLY the digit 1 or 0.\n\n"
        f"QUESTION: {question}\nEXPECTED: {expected}\nGENERATED: {generated}\n\n"
        "Grade (1 or 0):")
    return v, r


def judge_faithfulness(generated, context, image_paths=None):
    """
    Is every claim supported by the evidence the system actually used?

    Figure questions route to the vision judge with the image attached: asking a
    text-only model whether an answer is supported by a figure it cannot see
    would be guessing, and would silently corrupt the metric on 35% of items.
    """
    if image_paths:
        v, r = judge(VISION_JUDGE,
            "Check whether the ANSWER is faithful to the evidence: every claim "
            "supported by the attached image and/or the text context, nothing "
            "invented. Reply with ONLY the digit 1 or 0.\n\n"
            f"TEXT CONTEXT:\n{context}\n\nANSWER:\n{generated}\n\nGrade (1 or 0):",
            images=image_paths)
        return v, r

    v, r = judge(TEXT_JUDGE,
        "Check whether an answer is FAITHFUL to the context (every claim supported, "
        "nothing invented). Reply with ONLY the digit 1 or 0.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{generated}\n\nGrade (1 or 0):")
    return v, r


def judge_relevancy(question, generated):
    v, r = judge(TEXT_JUDGE,
        "Check whether an answer is RELEVANT to the question (addresses it, "
        "regardless of correctness). Reply with ONLY the digit 1 or 0.\n\n"
        f"QUESTION: {question}\nANSWER: {generated}\n\nGrade (1 or 0):")
    return v, r


def retrieval_metrics(ranked_sources, expected_source):
    """Recall@k and MRR against the expected source(s). ranked_sources is best-first.

    expected_source may be:
      str  — one relevant source (PubLayNet: one page holds the answer)
      list — several, ALL needed (HotpotQA: 2 gold paragraphs per question)

    For the list case this returns PARTIAL recall — the fraction of gold sources
    retrieved — because that is the quantity the KG story turns on. Getting one
    of two gold paragraphs is exactly the condition where a graph can bridge to
    the missing one; scoring it the same as getting neither would hide the
    mechanism. MRR uses the best-ranked gold source.
    """
    if isinstance(expected_source, str):
        if expected_source in ranked_sources:
            rank = ranked_sources.index(expected_source) + 1
            return 1, 1.0 / rank
        return 0, 0.0

    # Multi-source: partial credit.
    found = [s for s in expected_source if s in ranked_sources]
    if not found:
        return 0.0, 0.0
    recall = len(found) / len(expected_source)
    best_rank = min(ranked_sources.index(s) + 1 for s in found)
    return recall, 1.0 / best_rank


def retrieval_complete(ranked_sources, expected_source):
    """
    Did retrieval return EVERY gold source? 1/0.

    Tracked separately from partial recall because it splits the eval into the
    two conditions the KG claim depends on:

      complete=1 — the answer is fully in context; the KG should add nothing
                   (this is the PubLayNet situation, where +KG hurt)
      complete=0 — a gold paragraph is missing; ONLY a graph edge can reach it

    Reporting accuracy split on this flag tests the mechanism directly instead
    of inferring it from an aggregate.
    """
    if isinstance(expected_source, str):
        return 1 if expected_source in ranked_sources else 0
    return 1 if all(s in ranked_sources for s in expected_source) else 0


# ---------- the four systems ----------
# Each returns: (GenResult, text_sources_ranked, image_names_ranked_or_None,
#                context_used, image_paths_or_None)
# Text-only systems return None for the image ranking — by design they cannot
# retrieve images, so figure Recall is 0 for them.
def run_baseline(q, model, text_index, chunks, sources, graph, img_index,
                   img_names, captions, image_dir=IMAGE_DIR, vlm=True):
    retrieved = retrieve(q, text_index, chunks, sources, k=K)
    context = "\n\n".join(chunk for _, _, chunk in retrieved)
    return generate(q, retrieved, model=model), [s for _, s, _ in retrieved], None, context, None

def run_kg(q, model, text_index, chunks, sources, graph, img_index,
             img_names, captions, image_dir=IMAGE_DIR, vlm=True):
    retrieved = retrieve(q, text_index, chunks, sources, k=K)
    # KG facts are restricted to the pages retrieval returned. Without this,
    # common nodes ("patient", "women") match the query and pull unrelated facts
    # from across the corpus — measured to drop faithfulness 0.714 -> 0.571.
    allowed = {s for _, s, _ in retrieved}
    facts = graph_facts_for_query(q, graph, allowed_sources=allowed)
    context = "\n\n".join(chunk for _, _, chunk in retrieved) + "\n" + "\n".join(facts)
    return (generate_with_graph(q, retrieved, facts, model=model),
            [s for _, s, _ in retrieved], None, context, None)

def run_multimodal(q, model, text_index, chunks, sources, graph, img_index,
                     img_names, captions, image_dir=IMAGE_DIR, vlm=True):
    retrieved = retrieve(q, text_index, chunks, sources, k=K)

    # Retrieve deep, send shallow. `scored` is the full ranked list returned for
    # metrics; only a prefix of it ever reaches the generator.
    scored = retrieve_images(q, img_index, img_names, captions,
                             k=K_IMAGES_RETRIEVE)
    sent = scored[:K_IMAGES_PIXELS] if vlm else scored[:K_IMAGES_CAPTION]

    context = ("\n\n".join(chunk for _, _, chunk in retrieved) + "\n" +
               "\n".join(cap for _, _, cap in sent))
    paths = [os.path.join(image_dir, n) for _, n, _ in sent]
    return (generate_multimodal(q, retrieved, sent, model=model,
                                vlm=vlm, image_dir=image_dir),
            [s for _, s, _ in retrieved],
            [n for _, n, _ in scored],      # metrics see all K_IMAGES_RETRIEVE
            context, paths)

def run_both(q, model, text_index, chunks, sources, graph, img_index,
               img_names, captions, image_dir=IMAGE_DIR, vlm=True):
    retrieved = retrieve(q, text_index, chunks, sources, k=K)
    allowed = {s for _, s, _ in retrieved}
    facts = graph_facts_for_query(q, graph, allowed_sources=allowed)

    scored = retrieve_images(q, img_index, img_names, captions,
                             k=K_IMAGES_RETRIEVE)
    sent = scored[:K_IMAGES_PIXELS] if vlm else scored[:K_IMAGES_CAPTION]

    context = ("\n\n".join(chunk for _, _, chunk in retrieved) + "\n" + "\n".join(facts) +
               "\n" + "\n".join(cap for _, _, cap in sent))
    paths = [os.path.join(image_dir, n) for _, n, _ in sent]
    return (generate_full(q, retrieved, facts, sent, model=model,
                          vlm=vlm, image_dir=image_dir),
            [s for _, s, _ in retrieved],
            [n for _, n, _ in scored],      # metrics see all K_IMAGES_RETRIEVE
            context, paths)

ALL_SYSTEMS = {
    "baseline":    run_baseline,
    "+KG":         run_kg,
    "+multimodal": run_multimodal,
    "+both":       run_both,
}
# Default: the full four-way ablation (PubLayNet — has figures).
# --systems baseline,+KG runs the text-only pair (HotpotQA — no images, so the
# multimodal systems have nothing to retrieve and would only burn tokens).
DEFAULT_SYSTEMS = list(ALL_SYSTEMS)

# Quality metrics + the efficiency metrics the paper's cost/latency tables need.
# `complete` = did retrieval return ALL gold sources? Splitting accuracy on this
# is what tests the KG mechanism directly (see retrieval_complete).
METRICS = ("recall", "complete", "mrr", "acc", "faith", "rel",
           "in_tok", "out_tok", "latency_ms", "cost_usd")


def main():
    ap = argparse.ArgumentParser(description="Four-way ablation for one generator.")
    ap.add_argument("qfile", nargs="?", default="questions_publaynet_text.json")
    ap.add_argument("--model", default="gpt4o-mini", choices=GENERATORS + DIAGNOSTICS,
                    help="which generator to run (retrieval is identical for all). "
                         f"Diagnostic-only, excluded from the comparison: "
                         f"{', '.join(DIAGNOSTICS)}")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N questions (use for the pilot)")
    ap.add_argument("--systems", default=None,
                    help="comma-separated subset to run, e.g. 'baseline,+KG'. "
                         "Default: all four. Use the pair on image-free corpora "
                         f"like HotpotQA. Choices: {', '.join(ALL_SYSTEMS)}")
    ap.add_argument("--image-dir", default=IMAGE_DIR,
                    help="directory of figure crops (used by +multimodal/+both)")
    ap.add_argument("--corpus", default=None,
                    help="corpus directory, e.g. hotpotqa_corpus. Default: "
                         "rag_basics.CORPUS_DIR. Set this rather than editing "
                         "rag_basics.py — running one dataset's questions "
                         "against another's index looks plausible and is nonsense.")
    vlm_group = ap.add_mutually_exclusive_group()
    vlm_group.add_argument("--vlm", dest="vlm", action="store_true",
                           help="pass retrieved figures to the generator as PIXELS "
                                "(default)")
    vlm_group.add_argument("--no-vlm", dest="vlm", action="store_false",
                           help="caption-mediated: pass the figure's text caption "
                                "instead of the image. This was the variant used in "
                                "the original 100-page report (figure acc 0.514), so "
                                "it is the like-for-like comparison. Text-only models "
                                "a text-only model would use this path regardless.")
    ap.set_defaults(vlm=True)
    args = ap.parse_args()

    # Resolve which systems to run. Keep the registry's order so summary rows
    # are comparable across runs regardless of how --systems was written.
    if args.systems:
        wanted = [s.strip() for s in args.systems.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in ALL_SYSTEMS]
        if unknown:
            ap.error(f"unknown system(s): {', '.join(unknown)}. "
                     f"Choices: {', '.join(ALL_SYSTEMS)}")
        systems = [(n, ALL_SYSTEMS[n]) for n in ALL_SYSTEMS if n in wanted]
    else:
        systems = [(n, ALL_SYSTEMS[n]) for n in DEFAULT_SYSTEMS]

    model = args.model
    with open(args.qfile, encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]

    qset   = os.path.splitext(os.path.basename(args.qfile))[0]
    # The image-delivery path is part of the run's identity, not just a setting.
    #
    # Without this, `--no-vlm` writes summary_{qset}_{model}_*.csv — the SAME
    # name as the pixel run. run_matrix.py would then treat a completed caption
    # run as proof the VLM run was done and skip it, and collect_results.py,
    # which keeps the newest run_id per (question_set, model), would silently
    # drop whichever ran first. The pixels-vs-captions comparison would quietly
    # compare a run against itself.
    if not args.vlm:
        qset += "_captions"
    run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")   # colon-free for Windows

    cfg = MODELS[model]
    print(f"\nAblation — {qset} ({len(questions)} questions)")
    print(f"Generator : {model}  ({cfg['vendor']}, {cfg['access']}, "
          f"{'vision' if cfg['vision'] else 'TEXT-ONLY → caption-mediated'})")
    if cfg["access"] == "diagnostic":
        print(f"{'':12}DIAGNOSTIC RUN — excluded from the reported comparison")
    print(f"Systems   : {', '.join(n for n, _ in systems)}")
    if len(systems) < len(ALL_SYSTEMS):
        skipped = [n for n in ALL_SYSTEMS if n not in dict(systems)]
        print(f"{'':12}({', '.join(skipped)} skipped)")
    # Which image path is under test. Two runs of the same model on the same
    # questions differ only by this, so it has to be visible.
    if any(n in ("+multimodal", "+both") for n, _ in systems):
        variant = "pixels (VLM)" if args.vlm else "captions (caption-mediated)"
        print(f"Images    : {variant}")
        if args.vlm and not cfg["vision"]:
            print(f"{'':12}(--vlm has no effect: {model} is text-only)")
    print(f"Judges    : {TEXT_JUDGE} (acc/rel/text-faith), "
          f"{VISION_JUDGE} (figure faith)")
    print("=" * 78)

    # Build retrieval ONCE — shared across systems and generators, so the only
    # variable between runs is the generator.
    # Build retrieval ONCE — shared across systems and generators, so the only
    # variable between runs is the generator.
    corpus_dir = args.corpus or DEFAULT_CORPUS_DIR
    print(f"Corpus    : {corpus_dir}")
    text_index, chunks, sources = build_index(corpus_dir=corpus_dir)
    # with_sources=True carries each triple's origin page, so KG facts can be
    # restricted to what retrieval returned. Reuses the existing cache — no
    # re-extraction.
    graph = build_graph(build_triples(corpus_dir=corpus_dir, with_sources=True))

    # Only build the CLIP index if an image system is actually running. On an
    # image-free corpus (HotpotQA) there is nothing to embed, and captions.json
    # would not exist.
    needs_images = any(n in ("+multimodal", "+both") for n, _ in systems)
    if needs_images:
        img_index, img_names, captions = build_image_index()
    else:
        img_index, img_names, captions = None, [], {}
        print("Image index skipped (no image system selected).")

    agg = {name: {m: 0.0 for m in METRICS} for name, _ in systems}
    detail_rows = []
    n_errors = 0
    # Per-system, not just a global count. A rate limit does not fall evenly:
    # the pixel paths (+multimodal, +both) carry ~1,000 extra tokens per call,
    # so they burn a token-per-minute bucket first and fail far more often than
    # baseline. A global count hides that skew; these columns expose it.
    errs = {name: 0 for name, _ in systems}

    for i, item in enumerate(questions, 1):
        question, expected_src, expected_ans = item["q"], item["source"], item["answer"]
        qtype = item.get("type", "text")
        # HotpotQA questions carry a LIST of gold paragraphs; PubLayNet a single
        # string. Image questions are keyed off a .png source.
        srcs_list = expected_src if isinstance(expected_src, list) else [expected_src]
        is_image_question = any(s.endswith(".png") for s in srcs_list)
        row = {"question": question,
               "expected_source": "|".join(srcs_list),
               "question_type": qtype, "model": model}

        # Pixels go out only when the answer actually lives in a figure.
        #
        # Previously `vlm=args.vlm` was passed straight through, so on a TEXT
        # question +multimodal still CLIP-matched an image and sent the crop.
        # Measured on gemini-flash-lite / questions_publaynet_text: baseline
        # 484 in_tok, +multimodal 1,573 — an extra ~1,030 tokens of base64 for a
        # figure the question had nothing to do with (CLIP's top hit was a table
        # about dysphonia for a question about leptospirosis vaccines). It
        # tripled cost/100q and contaminated accuracy: +multimodal read 0.600 vs
        # baseline 0.514 on identical retrieval (0.914 across all systems), a
        # "gain" attributable to an image that was never meant to be there.
        #
        # On text/multi-hop questions +multimodal now means the caption block,
        # which is what the paper's Method section describes.
        use_vlm = args.vlm and is_image_question

        for name, run_system in systems:
            gen, text_ranked, img_ranked, context, img_paths = run_system(
                question, model, text_index, chunks, sources, graph,
                img_index, img_names, captions,
                image_dir=args.image_dir, vlm=use_vlm)

            if gen.error:
                n_errors += 1
                errs[name] += 1
                print(f"  {i:3d}. [{name}] ERROR: {gen.error}")

            # Score retrieval against whichever modality holds the answer.
            ranked = (img_ranked or []) if is_image_question else text_ranked
            rec, mrr = retrieval_metrics(ranked, expected_src)
            complete = retrieval_complete(ranked, expected_src)

            acc,   ja = judge_answer(question, expected_ans, gen.text)
            # Figure questions: judge faithfulness WITH the image attached.
            faith, jf = judge_faithfulness(
                gen.text, context,
                image_paths=img_paths if (is_image_question and img_paths) else None)
            rel,   jr = judge_relevancy(question, gen.text)

            agg[name]["recall"]     += rec
            agg[name]["complete"]   += complete
            agg[name]["mrr"]        += mrr
            agg[name]["acc"]        += acc
            agg[name]["faith"]      += faith
            agg[name]["rel"]        += rel
            # Efficiency: generation only. Judge cost is a measurement expense,
            # not a property of the system being measured.
            agg[name]["in_tok"]     += gen.in_tok
            agg[name]["out_tok"]    += gen.out_tok
            agg[name]["latency_ms"] += gen.latency_ms
            agg[name]["cost_usd"]   += gen.cost_usd

            row[f"{name}_acc"]        = acc
            row[f"{name}_faith"]      = faith
            row[f"{name}_rel"]        = rel
            row[f"{name}_rec"]        = round(rec, 3)
            # The split-analysis key: accuracy conditioned on this flag is what
            # shows whether the KG bridges gaps retrieval left.
            row[f"{name}_complete"]   = complete
            row[f"{name}_mrr"]        = round(mrr, 3)
            row[f"{name}_in_tok"]     = gen.in_tok
            row[f"{name}_out_tok"]    = gen.out_tok
            row[f"{name}_latency_ms"] = round(gen.latency_ms, 1)
            row[f"{name}_cost_usd"]   = round(gen.cost_usd, 6)
            row[f"{name}_answer"]     = gen.text.replace("\n", " ")[:160]
            row[f"{name}_error"]      = gen.error

        detail_rows.append(row)
        print(f"  {i:3d}/{len(questions)}  {question[:58]}")

    n = len(questions)

    # ---------- console summary ----------
    print("\n" + "=" * 78)
    print(f"QUALITY — {model}")
    print(f"{'metric':<14}" + "".join(f"{name:>15}" for name, _ in systems))
    print("-" * 78)
    for m, label in [("recall", f"Recall@{K}"), ("complete", "AllGoldFound"),
                     ("mrr", "MRR"),
                     ("acc", "AnswerAcc"), ("faith", "Faithfulness"),
                     ("rel", "AnswerRel")]:
        print(f"{label:<14}" + "".join(f"{agg[nm][m]/n:>15.3f}" for nm, _ in systems))

    print(f"\nEFFICIENCY — {model}")
    print(f"{'metric':<14}" + "".join(f"{name:>15}" for name, _ in systems))
    print("-" * 78)
    print(f"{'in_tok/q':<14}"   + "".join(f"{agg[nm]['in_tok']/n:>15.0f}"  for nm, _ in systems))
    print(f"{'out_tok/q':<14}"  + "".join(f"{agg[nm]['out_tok']/n:>15.0f}" for nm, _ in systems))
    print(f"{'latency ms':<14}" + "".join(f"{agg[nm]['latency_ms']/n:>15.0f}" for nm, _ in systems))
    print(f"{'cost/100q $':<14}"+ "".join(f"{agg[nm]['cost_usd']/n*100:>15.4f}" for nm, _ in systems))

    # ---------- summary CSV ----------
    # A run with failed calls is QUARANTINED: the file is written (the data is
    # real and worth inspecting) but named summary_FAILED_*, so it does not match
    # the summary_{qset}_{model}_*.csv glob that run_matrix.py uses to detect
    # completed runs or that collect_results.py folds into the paper's tables.
    #
    # Why this matters: llm_client returns GenResult(text="", error=...) after
    # exhausting retries, and the judges score "" as 0. So a Groq 429 storm
    # produces a summary full of plausible-looking zeros — and because pixel
    # paths burn tokens fastest, the zeros land disproportionately on
    # +multimodal and +both. That reads as "vision paths underperform", which is
    # a rate limit wearing the costume of a finding.
    quarantined = n_errors > 0
    prefix = "summary_FAILED" if quarantined else "summary"
    summary_csv = os.path.join(RESULTS_DIR, f"{prefix}_{qset}_{model}_{stamp}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "run_id", "question_set", "question_type", "model", "vendor", "access",
            "n_questions", "system",
            "recall", "complete", "mrr", "acc", "faith", "rel",
            "mean_in_tok", "mean_out_tok", "mean_latency_ms",
            "total_cost_usd", "cost_per_100q_usd",
            # A failed call returns GenResult(text="", error=...). The judge
            # then scores "" as 0, so a rate-limited cell is indistinguishable
            # from a real null in this file. This column is the difference.
            "n_errors", "error_rate"])
        w.writeheader()
        for name, _ in systems:
            a = agg[name]
            w.writerow({
                "run_id": run_id, "question_set": qset,
                "question_type": questions[0].get("type", "text"),
                "model": model, "vendor": cfg["vendor"], "access": cfg["access"],
                "n_questions": n, "system": name,
                "recall": round(a["recall"]/n, 3),
                "complete": round(a["complete"]/n, 3),
                "mrr":    round(a["mrr"]/n, 3),
                "acc":    round(a["acc"]/n, 3),
                "faith":  round(a["faith"]/n, 3),
                "rel":    round(a["rel"]/n, 3),
                "mean_in_tok":     round(a["in_tok"]/n, 1),
                "mean_out_tok":    round(a["out_tok"]/n, 1),
                "mean_latency_ms": round(a["latency_ms"]/n, 1),
                "total_cost_usd":  round(a["cost_usd"], 6),
                "cost_per_100q_usd": round(a["cost_usd"]/n*100, 4),
                "n_errors": errs[name],
                "error_rate": round(errs[name]/n, 3),
            })

    detail_csv = os.path.join(RESULTS_DIR, f"detail_{qset}_{model}_{stamp}.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    print(f"\nSummary -> {summary_csv}")
    print(f"Detail  -> {detail_csv}")
    if n_errors:
        print("\n" + "!" * 78)
        print(f"  QUARANTINED: {n_errors} generation call(s) failed after retries.")
        print(f"  Failures by system (a failed call is scored as 0, not skipped):")
        for name, _ in systems:
            if errs[name]:
                print(f"    {name:<14} {errs[name]:>3}/{n} questions "
                      f"({errs[name]/n:.0%})")
        print()
        print(f"  Summary written as {os.path.basename(summary_csv)} — the FAILED_")
        print(f"  prefix keeps it OUT of run_matrix.py's completed-run detection")
        print(f"  and OUT of collect_results.py's tables. Fix the cause and re-run;")
        print(f"  do NOT rename this file to make it count.")
        print("!" * 78)
    if not cfg["vision"]:
        print(f"\nNote: {model} is text-only, so +multimodal/+both used the "
              f"caption-mediated variant.")
    print("Note: text-only systems (baseline, +KG) score 0 retrieval on figure "
          "questions by design — they cannot retrieve images.")


if __name__ == "__main__":
    main()
