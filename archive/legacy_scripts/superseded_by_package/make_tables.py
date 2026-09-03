"""
make_tables.py — regenerates every table in the paper from results/*.csv.

Run from the repo root:
    python make_tables.py                      # reads results/, writes paper_tables/
    python make_tables.py --results results --out paper_tables

Selection rule: for each (question_set, model), use one newest valid run as a
unit. Cells are never backfilled from another timestamp. Release artifacts
should use the frozen-manifest path exposed by ``rag artifacts``.

SPIQA figure tiering:
  * 5 caption-contaminated crops are excluded everywhere (verified leak).
  * If spiqa_corpus/ is present, the remaining questions are further split into
    text-recoverable (answer string appears in the source paper's corpus text)
    and strict-pixel tiers. Absent the corpus, only the caption exclusion is
    applied and a note is printed.

Outputs (markdown + csv): T1 graph three-way, T2 bridge/comparison mechanism,
T3 evidence-conditioned accuracy & leakage, T4 figure-question integrity,
T5 full four-way ablation per corpus, T6 efficiency.

Stdlib only.
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

MODELS = ["gpt4o-mini", "gemini-flash-lite", "llama4-scout", "llama4-maverick"]
DIAG = ["gpt4o"]
SYSTEMS = ["baseline", "+KG", "+KGret", "+multimodal", "+both"]

CONTAMINATED_SPIQA_FIGS = {
    "1809.00263v5-Figure12-1.png",
    "1906.10843v1-Figure5-1.png",
    "1705.02946v3-Figure6-1.png",
    "1803.01128v3-Table2-1.png",
    "1802.07351v2-Figure10-1.png",
}

GRAPH_SETS = [
    "publaynet_multihop",
    "spiqa_multihop",
    "spiqa_multihop_cross",
    "hotpotqa_bridge",
    "hotpotqa_comparison",
]
SET_LABEL = {
    "publaynet_multihop": "PubLayNet multi-hop (within-page)",
    "spiqa_multihop": "SPIQA multi-hop (within-paper)",
    "spiqa_multihop_cross": "SPIQA multi-hop (cross-paper)",
    "hotpotqa_bridge": "HotpotQA bridge",
    "hotpotqa_comparison": "HotpotQA comparison (control)",
}

FNAME_RE = re.compile(
    r"(summary|detail)_questions_(.+)_"
    r"(gpt4o-mini|gemini-flash-lite|llama4-maverick|llama4-scout|gpt4o)_"
    r"(\d{8}_\d{6})\.csv$"
)


def norm(s):
    return re.sub(r"[^a-z0-9.]+", " ", s.lower()).strip()


def load_runs(results_dir):
    """-> runs[(qset, model)][stamp] = {'summary': rows, 'detail_path': path}"""
    runs = defaultdict(dict)
    for path in glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        kind, qset, model, stamp = m.groups()
        entry = runs[(qset, model)].setdefault(stamp, {})
        if kind == "summary":
            with open(path, encoding="utf-8") as f:
                entry["summary"] = list(csv.DictReader(f))
        else:
            entry["detail_path"] = path
    return runs


def valid(summary_rows, qset):
    """Wrong-corpus guard: text-gold sets with baseline recall 0 are invalid."""
    if "figures" in qset:
        return True
    for r in summary_rows:
        if r["system"] == "baseline":
            return float(r["recall"]) > 0.0
    return True


def select(runs):
    """cell[(qset, model, system)] = (summary_row, detail_path, stamp);
    select one newest valid run per question-set/model without backfilling."""
    cell = {}
    for (qset, model), by_stamp in runs.items():
        for stamp in sorted(by_stamp, reverse=True):
            entry = by_stamp[stamp]
            rows = entry.get("summary", [])
            if not rows or not valid(rows, qset):
                continue
            for r in rows:
                key = (qset, model, r["system"])
                cell[key] = (r, entry.get("detail_path"), stamp)
            break
    return cell


def detail_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x, nd=3):
    return f"{float(x):.{nd}f}"


def write_table(out_dir, name, header, rows, note=None):
    md = os.path.join(out_dir, f"{name}.md")
    cv = os.path.join(out_dir, f"{name}.csv")
    with open(cv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join("---" for _ in header) + "|\n")
        f.writelines("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)
        if note:
            f.write(f"\n_{note}_\n")
    print(f"  wrote {name} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join("artifacts", "runs"))
    ap.add_argument("--out", default=os.path.join("artifacts", "tables"))
    ap.add_argument("--spiqa-corpus", default="spiqa_corpus")
    ap.add_argument(
        "--spiqa-figq",
        default=os.path.join("data", "questions", "questions_spiqa_figures.json"),
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = load_runs(args.results)
    cell = select(runs)

    got = defaultdict(set)
    for qset, model, system in cell:
        got[qset].add(model)
    print("Coverage:")
    for qset in sorted(got):
        missing = [m for m in MODELS if m not in got[qset]]
        line = f"  {qset:<28} models: {len(got[qset] & set(MODELS))}/4"
        if missing:
            line += f"   MISSING: {', '.join(missing)}"
        print(line)

    # ── T1: graph three-way, acc (faith) per model ──────────────────────────
    header = ["question set", "system", "AllGoldFound"] + MODELS
    rows = []
    for qset in GRAPH_SETS:
        for system in ["baseline", "+KG", "+KGret"]:
            vals, comp = [], None
            for model in MODELS:
                c = cell.get((qset, model, system))
                if c:
                    r = c[0]
                    vals.append(f"{fmt(r['acc'])} ({fmt(r['faith'])})")
                    comp = r["complete"]
                else:
                    vals.append("—")
            if any(v != "—" for v in vals):
                rows.append([SET_LABEL[qset], system, comp or "—"] + vals)
    write_table(
        args.out,
        "T1_graph_three_way",
        header,
        rows,
        "Cells: accuracy (faithfulness). AllGoldFound is retrieval "
        "completeness over ALL gold sources; identical across models "
        "within a system because the retrieval stack is fixed. +KGret "
        "expands the candidate set, so its completeness differs by design.",
    )

    # ── T2: bridge vs comparison mechanism (mean over the 4 generators) ─────
    header = [
        "set",
        "system",
        "AllGoldFound",
        "mean acc",
        "mean faith",
        "Δacc vs baseline",
    ]
    rows = []
    for qset in ["hotpotqa_bridge", "hotpotqa_comparison"]:
        base_acc = None
        for system in ["baseline", "+KG", "+KGret"]:
            accs, faiths, comp = [], [], None
            for model in MODELS:
                c = cell.get((qset, model, system))
                if c:
                    accs.append(float(c[0]["acc"]))
                    faiths.append(float(c[0]["faith"]))
                    comp = c[0]["complete"]
            if not accs:
                continue
            macc = sum(accs) / len(accs)
            if system == "baseline":
                base_acc = macc
            rows.append(
                [
                    SET_LABEL[qset],
                    system,
                    comp,
                    fmt(macc),
                    fmt(sum(faiths) / len(faiths)),
                    f"{macc - base_acc:+.3f}" if base_acc is not None else "—",
                ]
            )
    write_table(
        args.out,
        "T2_bridge_vs_comparison",
        header,
        rows,
        "The mechanism table: bridge questions have low baseline "
        "completeness (the need), comparison questions retrieve fine "
        "(the control). Means over all four generators.",
    )

    # ── T3: evidence-conditioned accuracy + parametric leakage ──────────────
    header = [
        "question set",
        "system",
        "acc | complete",
        "acc | incomplete",
        "n complete",
        "n incomplete",
    ]
    rows = []
    for qset in GRAPH_SETS:
        for system in ["baseline", "+KG", "+KGret"]:
            c1 = c0 = a1 = a0 = 0
            for model in MODELS:
                c = cell.get((qset, model, system))
                if not c or not c[1]:
                    continue
                for r in detail_rows(c[1]):
                    acc_col, comp_col = f"{system}_acc", f"{system}_complete"
                    if acc_col not in r or r[acc_col] in ("", None):
                        continue
                    comp = float(r.get(comp_col, 0) or 0) >= 1.0
                    acc = int(float(r[acc_col]))
                    if comp:
                        c1 += 1
                        a1 += acc
                    else:
                        c0 += 1
                        a0 += acc
            if c1 + c0:
                rows.append(
                    [
                        SET_LABEL[qset],
                        system,
                        fmt(a1 / c1) if c1 else "—",
                        fmt(a0 / c0) if c0 else "—",
                        c1,
                        c0,
                    ]
                )
    write_table(
        args.out,
        "T3_evidence_conditioned",
        header,
        rows,
        "Pooled over generators. 'acc | incomplete' for baseline is the "
        "parametric-leakage floor: correct answers produced without "
        "complete gold evidence in context.",
    )

    # ── T4: figure-question integrity ───────────────────────────────────────
    tiers = {}
    if os.path.exists(args.spiqa_figq):
        with open(args.spiqa_figq, encoding="utf-8") as f:
            figq = json.load(f)
        corpus_ok = os.path.isdir(args.spiqa_corpus)
        for q in figq:
            src = q["source"]
            if src in CONTAMINATED_SPIQA_FIGS:
                tiers[src] = "caption-contaminated"
            elif corpus_ok:
                paper = os.path.join(args.spiqa_corpus, src.split("-")[0] + ".txt")
                intext = False
                if os.path.exists(paper):
                    with open(paper, encoding="utf-8") as pf:
                        intext = len(norm(q["answer"])) >= 2 and norm(
                            q["answer"]
                        ) in norm(pf.read())
                tiers[src] = "text-recoverable" if intext else "strict-pixel"
            else:
                tiers[src] = "clean (corpus absent: tier unknown)"
        if not corpus_ok:
            print(
                f"  NOTE: {args.spiqa_corpus}/ not found — SPIQA figure "
                "tiering limited to the caption exclusion."
            )

    header = ["set / tier", "n", "model", "text-only baseline acc", "+multimodal acc"]
    rows = []
    for qset in ["publaynet_figures", "publaynet_figures_caption"]:
        for model in MODELS:
            cb = cell.get((qset, model, "baseline"))
            cm = cell.get((qset, model, "+multimodal"))
            if cb:
                rows.append(
                    [
                        qset,
                        cb[0]["n_questions"],
                        model,
                        fmt(cb[0]["acc"]),
                        fmt(cm[0]["acc"]) if cm else "—",
                    ]
                )
    tier_names = sorted({t for t in tiers.values()})
    for model in MODELS:
        c = cell.get(("spiqa_figures", model, "baseline"))
        cm = cell.get(("spiqa_figures", model, "+multimodal"))
        if not c or not c[1]:
            continue
        det = {r["expected_source"]: r for r in detail_rows(c[1])}
        for tier in tier_names:
            srcs = [s for s, t in tiers.items() if t == tier and s in det]
            if not srcs:
                continue
            b = sum(int(float(det[s]["baseline_acc"])) for s in srcs)
            mm = (
                sum(int(float(det[s]["+multimodal_acc"])) for s in srcs)
                if cm and "+multimodal_acc" in next(iter(det.values()))
                else None
            )
            rows.append(
                [
                    f"spiqa_figures [{tier}]",
                    len(srcs),
                    model,
                    f"{b}/{len(srcs)}",
                    f"{mm}/{len(srcs)}" if mm is not None else "—",
                ]
            )
    write_table(
        args.out,
        "T4_figure_integrity",
        header,
        rows,
        "SPIQA figures tiered by leak path: caption-contaminated "
        "(excluded from headline claims), text-recoverable (answer "
        "present in corpus prose — text-only success expected), "
        "strict-pixel (the zero-claim tier; residual correct answers "
        "are parametric).",
    )

    # ── T5: full four-way ablation per corpus (Table II + siblings) ─────────
    header = [
        "question set",
        "model",
        "system",
        "recall",
        "complete",
        "acc",
        "faith",
        "rel",
    ]
    rows = []
    for (qset, model, system), (r, _, _) in sorted(cell.items()):
        if model in DIAG:
            continue
        rows.append(
            [
                qset,
                model,
                system,
                r["recall"],
                r["complete"],
                r["acc"],
                r["faith"],
                r["rel"],
            ]
        )
    write_table(
        args.out,
        "T5_all_runs_flat",
        header,
        rows,
        "Every selected (newest valid) cell, flat — the source of "
        "truth behind T1–T4 and the per-corpus four-way tables.",
    )

    # ── T6: efficiency ──────────────────────────────────────────────────────
    header = [
        "question set",
        "model",
        "system",
        "in_tok/q",
        "out_tok/q",
        "latency ms",
        "cost/100q $",
    ]
    rows = []
    for qset in ["publaynet_figures", "spiqa_multihop_cross", "hotpotqa_bridge"]:
        for model in MODELS:
            for system in SYSTEMS:
                c = cell.get((qset, model, system))
                if c:
                    r = c[0]
                    rows.append(
                        [
                            qset,
                            model,
                            system,
                            r["mean_in_tok"],
                            r["mean_out_tok"],
                            r["mean_latency_ms"],
                            r["cost_per_100q_usd"],
                        ]
                    )
    write_table(args.out, "T6_efficiency", header, rows)

    print(f"\nAll tables -> {args.out}/  (markdown + csv)")


if __name__ == "__main__":
    main()
