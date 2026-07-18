"""
collect_results.py — concatenates every summary CSV in results/ into one table.

The summary CSVs are already long-format (one row per system) with quality AND
efficiency columns, so this mostly just stacks them, de-duplicates re-runs, and
emits the slices the paper needs.

Emits into results/:
  matrix_all.csv        every run, every system — the master table
  table_quality.csv     qset x model x system -> recall/mrr/acc/faith/rel
  table_efficiency.csv  qset x model x system -> tokens/latency/cost
  table_figures.csv     figure questions only, all models — the model comparison
  table_kg_delta.csv    +KG minus baseline, per qset x model — the KG null

De-duplication: keeps only the newest run_id per (question_set, model) pair,
so re-runs never double-count.

Usage:  python collect_results.py
        python collect_results.py --keep-all    (don't de-duplicate)
"""
import argparse, csv, glob, os, sys

RESULTS = "results"

QUALITY_COLS = ["recall", "complete", "mrr", "acc", "faith", "rel"]
EFFIC_COLS   = ["mean_in_tok", "mean_out_tok", "mean_latency_ms",
                "total_cost_usd", "cost_per_100q_usd"]
KEY_COLS     = ["question_set", "question_type", "model", "vendor",
                "access", "n_questions", "system"]

# The figure sets are two DIFFERENT experiments, not one set run twice:
#   *_figures          answers are cell values; 0/35 findable in captions
#   *_figures_caption   answers are prose; 25/34 findable in captions
# and the _captions suffix marks a --no-vlm (caption-mediated) run of either.
QSET_ORDER  = ["questions_publaynet_text",
               "questions_publaynet_text_captions",
               "questions_publaynet_multihop",
               "questions_publaynet_multihop_captions",
               "questions_publaynet_figures_caption",
               "questions_publaynet_figures_caption_captions",
               "questions_publaynet_figures",
               "questions_publaynet_figures_captions"]
MODEL_ORDER = ["gpt4o-mini", "gemini-flash-lite", "llama4-scout",
               "llama4-maverick", "gpt4o"]
SYS_ORDER   = ["baseline", "+KG", "+multimodal", "+both"]


def order_key(val, order):
    return (order.index(val) if val in order else len(order), str(val))


def load_all():
    rows = []
    files = sorted(glob.glob(os.path.join(RESULTS, "summary_*.csv")))
    # Quarantined runs (any generation call failed after retries) are written by
    # compare_all.py as summary_FAILED_*. They must never reach a paper table: a
    # failed call scores 0, so a rate-limited run looks like a real null — and
    # because pixel paths burn tokens fastest, those zeros cluster on
    # +multimodal/+both and read as a vision finding.
    failed = [f for f in files if os.path.basename(f).startswith("summary_FAILED")]
    files  = [f for f in files if not os.path.basename(f).startswith("summary_FAILED")]
    if failed:
        print(f"EXCLUDED {len(failed)} quarantined run(s) — re-run these:")
        for f in failed:
            print(f"  {os.path.basename(f)}")
        print()
    if not files:
        print(f"no usable summary CSVs in {RESULTS}\\")
        sys.exit(1)
    for p in files:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_source"] = os.path.basename(p)
                rows.append(r)
    print(f"read {len(files)} files, {len(rows)} system-rows")
    return rows


def dedupe(rows):
    """keep only the newest run_id per (question_set, model)"""
    newest = {}
    for r in rows:
        k = (r["question_set"], r["model"])
        if k not in newest or r["run_id"] > newest[k]:
            newest[k] = r["run_id"]
    kept = [r for r in rows
            if r["run_id"] == newest[(r["question_set"], r["model"])]]
    dropped = len(rows) - len(kept)
    if dropped:
        print(f"de-duplicated: dropped {dropped} rows from superseded runs")
    return kept


def fnum(r, col):
    try:
        return float(r[col])
    except (KeyError, ValueError, TypeError):
        return None


def write(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}   ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-all", action="store_true",
                    help="do not de-duplicate superseded runs")
    args = ap.parse_args()

    rows = load_all()
    if not args.keep_all:
        rows = dedupe(rows)

    rows.sort(key=lambda r: (order_key(r["question_set"], QSET_ORDER),
                             order_key(r["model"], MODEL_ORDER),
                             order_key(r["system"], SYS_ORDER)))

    print()
    all_cols = ["run_id"] + KEY_COLS + QUALITY_COLS + EFFIC_COLS + ["_source"]
    write(os.path.join(RESULTS, "matrix_all.csv"), all_cols, rows)

    write(os.path.join(RESULTS, "table_quality.csv"),
          ["question_type", "question_set", "model", "vendor", "access",
           "system"] + QUALITY_COLS, rows)

    write(os.path.join(RESULTS, "table_efficiency.csv"),
          ["question_type", "question_set", "model", "system"] + EFFIC_COLS,
          rows)

    figs = [r for r in rows if r.get("question_type") == "figure"]
    if figs:
        write(os.path.join(RESULTS, "table_figures.csv"),
              ["model", "vendor", "access", "system"] + QUALITY_COLS
              + ["mean_in_tok", "cost_per_100q_usd"], figs)

    # ---- KG delta: the null result, quantified ----
    by = {(r["question_set"], r["model"], r["system"]): r for r in rows}
    deltas = []
    for (qset, model, system), r in by.items():
        if system != "baseline":
            continue
        for target in ("+KG", "+both"):
            t = by.get((qset, model, target))
            if not t:
                continue
            base_acc, targ_acc = fnum(r, "acc"), fnum(t, "acc")
            base_tok, targ_tok = fnum(r, "mean_in_tok"), fnum(t, "mean_in_tok")
            if None in (base_acc, targ_acc):
                continue
            deltas.append({
                "question_type": r.get("question_type", ""),
                "question_set": qset,
                "model": model,
                "contrast": f"{target} - baseline",
                "acc_baseline": base_acc,
                "acc_target": targ_acc,
                "acc_delta": round(targ_acc - base_acc, 4),
                "in_tok_delta": (round(targ_tok - base_tok, 1)
                                 if None not in (base_tok, targ_tok) else ""),
            })
    if deltas:
        deltas.sort(key=lambda d: (order_key(d["question_set"], QSET_ORDER),
                                   order_key(d["model"], MODEL_ORDER),
                                   d["contrast"]))
        write(os.path.join(RESULTS, "table_kg_delta.csv"),
              ["question_type", "question_set", "model", "contrast",
               "acc_baseline", "acc_target", "acc_delta", "in_tok_delta"],
              deltas)

    # ---- console grids ----
    def grid(metric, fmt, label):
        print("\n" + "=" * 76)
        print(label)
        print("=" * 76)
        for qset in QSET_ORDER:
            models = sorted({r["model"] for r in rows if r["question_set"] == qset},
                            key=lambda m: order_key(m, MODEL_ORDER))
            if not models:
                continue
            print(f"\n{qset}")
            print(f"  {'model':<20}" + "".join(f"{s:>14}" for s in SYS_ORDER))
            print("  " + "-" * (20 + 14 * len(SYS_ORDER)))
            for model in models:
                cells = ""
                for s in SYS_ORDER:
                    r = by.get((qset, model, s))
                    v = fnum(r, metric) if r else None
                    cells += (format(v, fmt).rjust(14) if v is not None
                              else "".rjust(14))
                print(f"  {model:<20}{cells}")

    grid("acc", ">.3f", "AnswerAcc")
    grid("cost_per_100q_usd", ">.4f", "cost / 100q  ($)")
    grid("mean_in_tok", ">.0f", "input tokens / question")
    grid("mean_latency_ms", ">.0f", "latency (ms)")

    # ---- missing cells ----
    want_models = MODEL_ORDER[:4]
    # Only the core 12 are required. The caption-ceiling and pixels-vs-captions
    # runs are supplementary; absent cells there are not gaps in the matrix.
    CORE = ["questions_publaynet_text", "questions_publaynet_multihop",
            "questions_publaynet_figures"]
    missing = [(q, m) for q in CORE for m in want_models
               if not any(r["question_set"] == q and r["model"] == m
                          for r in rows)]
    if missing:
        print("\n" + "=" * 76)
        print(f"MISSING CORE CELLS ({len(missing)} of {len(CORE)*len(want_models)})")
        print("=" * 76)
        for q, m in missing:
            print(f"  {m:<20} {q}")
        print("\n  -> python run_matrix.py")
    else:
        print("\nmatrix complete — all 12 cells present.")


if __name__ == "__main__":
    main()
