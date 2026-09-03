"""
make_figures.py — generates Figures 2 and 3 for the paper as IEEE-width PDFs.

  figs/fig2_cost_accuracy.pdf   cost vs accuracy on pixel-only figure questions
  figs/fig3_image_tokens.pdf    input tokens/question, +multimodal, by generator

Reads results/table_quality.csv and results/table_efficiency.csv, joining on
(question_set, model, system).

Scope: the four PRODUCTION generators only. GPT-4o is excluded — it is a
diagnostic reference at 16x the input price of GPT-4o-mini, and putting it in
the same panel invites the reader to compare it as a fifth system. Its numbers
belong in the Table I diagnostic block, or in a separate panel if a
price-tier experiment is added later.

NOTE: deliberately does NOT read table_figures.csv. That file merges the
caption-answerable and pixel-only figure sets with no question_set column to
tell them apart — every model appears twice per system with different numbers.
Using it would silently plot one set's accuracy against the other's cost.

Usage:  python make_figures.py
Deps:   pip install matplotlib
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join("artifacts", "runs")
FIGS = os.path.join("artifacts", "figures")
PIXEL_ONLY = "questions_publaynet_figures"  # answers are NOT recoverable from captions

# The no-image reference line in Fig 3. Deliberately NOT red — red reads as
# "bad" or "threshold breached", and this is a neutral reference point, not a
# limit anyone is violating. Blue is outside the four-model palette (teal/green
# = open, orange/magenta = closed), so it cannot be mistaken for a data series.
BASELINE_COLOUR = "#1f77b4"

# One colour per model (ColorBrewer Dark2), consistent across both figures so a
# reader can carry the mapping from one panel to the other.
#
# Hues are grouped by access tier rather than assigned arbitrarily: the two open
# models are greens, the two closed models are warm. The open/closed split is an
# axis of the comparison, so the figure should read before the legend does.
#
# Dark2 over the pastel Accent palette because these are ~70pt markers on white,
# not filled regions. Accent's #ffff99 sits at ~97% luminance and effectively
# disappeared against the page, leaving the black edge to do all the work; Dark2
# runs 40-60% luminance and survives greyscale printing.
MODELS = [
    # key,                label,                   colour,    marker
    ("llama4-scout", "Llama 4 Scout", "#1b9e77", "o"),  # open, teal
    ("llama4-maverick", "Llama 4 Maverick", "#66a61e", "o"),  # open, green
    ("gemini-flash-lite", "Gemini 3.1 Flash-Lite", "#d95f02", "s"),  # closed, orange
    ("gpt4o-mini", "GPT-4o-mini", "#e7298a", "s"),  # closed, magenta
]

# IEEE single column is 3.5in. Fonts at 7-8pt so they match caption size once
# LaTeX places the figure at column width.
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(r, col):
    try:
        return float(r[col])
    except (KeyError, ValueError, TypeError):
        return None


def index_by(rows, qset, system):
    return {
        r["model"]: r
        for r in rows
        if r.get("question_set") == qset and r.get("system") == system
    }


def main():
    os.makedirs(FIGS, exist_ok=True)
    qual = load(os.path.join(RESULTS, "table_quality.csv"))
    effi = load(os.path.join(RESULTS, "table_efficiency.csv"))

    q_mm = index_by(qual, PIXEL_ONLY, "+multimodal")
    e_mm = index_by(effi, PIXEL_ONLY, "+multimodal")

    missing = [m for m, _, _, _ in MODELS if m not in q_mm or m not in e_mm]
    if missing:
        print(f"WARNING: no +multimodal row on {PIXEL_ONLY} for: {missing}")

    # ---------------------------------------------------------------- Fig 2
    # Cost vs accuracy. Log-x because the story is the RATIO between points:
    # on a linear axis the three cheap models collapse into the left margin.
    # Legend rather than inline labels — Maverick ($0.0189) and Gemini ($0.0171)
    # are within 10% on cost, so their annotations overlapped.
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    for key, label, colour, marker in MODELS:
        if key not in q_mm or key not in e_mm:
            continue
        x, y = fnum(e_mm[key], "cost_per_100q_usd"), fnum(q_mm[key], "acc")
        if x is None or y is None:
            continue
        ax.scatter(
            x,
            y,
            s=70,
            c=colour,
            marker=marker,
            edgecolors="black",
            linewidths=0.7,
            label=label,
            zorder=3,
        )

    # The retrieval ceiling: a generator cannot read a figure that was never
    # retrieved, so Recall@3 caps accuracy regardless of model. Every point
    # sitting far below a cap that is itself only 0.371 IS the finding.
    ceil = fnum(q_mm.get("gpt4o-mini", {}), "recall")
    if ceil:
        ax.axhline(ceil, ls="--", lw=0.9, c="#555555", zorder=1)
        ax.annotate(
            f"retrieval ceiling (Recall@3 = {ceil:.3f})",
            (0.0042, ceil - 0.028),
            fontsize=6.5,
            color="#444444",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Cost per 100 questions (USD, log scale)")
    ax.set_ylabel("Answer accuracy")
    ax.set_ylim(0, (ceil or 0.4) + 0.20)  # headroom for the legend
    ax.set_xlim(0.004, 0.30)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5, zorder=0)
    # Legend in the upper-right: all four points sit low (acc <= 0.114) and
    # left-to-middle, so that corner is empty. "center right" overlapped them.
    ax.legend(
        loc="upper right",
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.3,
        labelspacing=0.35,
    )

    out2 = os.path.join(FIGS, "fig2_cost_accuracy.pdf")
    fig.savefig(out2)
    plt.close(fig)
    print(f"wrote {out2}")

    # ---------------------------------------------------------------- Fig 3
    # Input tokens on +multimodal. Same figure, same text context, every model:
    # the spread is image tokenisation alone. Log-y — 861 vs 9477 is 11x.
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    labels, vals, cols = [], [], []
    for key, label, colour, _ in MODELS:
        if key not in e_mm:
            continue
        v = fnum(e_mm[key], "mean_in_tok")
        if v is None:
            continue
        labels.append(
            label.replace(" 3.1 Flash-Lite", "\n3.1 Flash-Lite").replace(
                "Llama 4 ", "Llama 4\n"
            )
        )
        vals.append(v)
        cols.append(colour)

    bars = ax.bar(
        range(len(vals)),
        vals,
        color=cols,
        edgecolor="black",
        linewidth=0.7,
        width=0.6,
        zorder=3,
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v * 1.07,
            f"{v:,.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            zorder=4,
        )

    # Reference line: what the SAME prompt costs with no image attached.
    # Across models this is 484-517 tokens (a 1.07x spread — same text context,
    # marginally different tokenisers), so a single line is a fair summary. The
    # gap between each bar and this line is the price of the image.
    e_base = index_by(effi, PIXEL_ONLY, "baseline")
    bv = [fnum(e_base[k], "mean_in_tok") for k, _, _, _ in MODELS if k in e_base]
    bv = [v for v in bv if v]
    if bv:
        lo, hi, mean = min(bv), max(bv), sum(bv) / len(bv)
        ax.axhline(mean, ls="--", lw=1.3, c=BASELINE_COLOUR, zorder=2)
        # Annotation stays top-left in clear space; an arrow connects it to the
        # line rather than moving the text down among the bars. A grey dashed
        # line at 0.9pt read as chart furniture, not as a reference the reader
        # was meant to use.
        ax.annotate(
            f"text-only baseline\n({lo:.0f}\u2013{hi:.0f} tok, all models)",
            xy=(0.55, mean),
            xytext=(-0.42, 11000),
            fontsize=6.5,
            color=BASELINE_COLOUR,
            ha="left",
            va="top",
            linespacing=1.3,
            arrowprops={
                "arrowstyle": "->",
                "color": BASELINE_COLOUR,
                "lw": 0.9,
                "shrinkA": 2,
                "shrinkB": 2,
                "connectionstyle": "arc3,rad=-0.15",
            },
        )

    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Input tokens / question (log scale)")
    ax.set_ylim(300, 22000)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5, zorder=0)

    out3 = os.path.join(FIGS, "fig3_image_tokens.pdf")
    fig.savefig(out3)
    plt.close(fig)
    print(f"wrote {out3}")

    # ---------------------------------------------------------------- report
    print("\nPlotted (pixel-only figure set, +multimodal):")
    print(f"  {'model':<20}{'cost/100q':>11}{'acc':>8}{'in_tok':>9}")
    for key, _, _, _ in MODELS:
        if key in q_mm and key in e_mm:
            print(
                f"  {key:<20}{fnum(e_mm[key], 'cost_per_100q_usd'):>11.4f}"
                f"{fnum(q_mm[key], 'acc'):>8.3f}{fnum(e_mm[key], 'mean_in_tok'):>9.0f}"
            )
    print("\nGPT-4o excluded (diagnostic; see Table I).")

    if os.path.isdir(TABLES):
        fig4_stage()
        fig5_leakage()
    else:
        print(f"NOTE: {TABLES}/ not found -- run make_tables.py first for Figs 4-5.")


# ---------------------------------------------------------------- Fig 4 / Fig 5
# The two stage-comparison figures. These read paper_tables/ (produced by
# make_tables.py) rather than results/, because they aggregate across corpora.
# System colours (not model colours): the contrast being drawn is between the
# two graph deployments, so +KG and +KGret each get one hue and the models are
# distinguished on the x-axis. Neither hue appears in the model palette above,
# so the mapping cannot be confused across figures.
KG_COLOUR = "#c8553d"  # generation-stage
KGRET_COLOUR = "#2e6f95"  # retrieval-stage
TABLES = os.path.join("artifacts", "tables")


def fig4_stage():
    """Per-generator accuracy deltas of +KG and +KGret over baseline on the
    two evidence-deficient sets. The asymmetry IS the finding: +KG scatters
    around zero, +KGret is uniformly positive."""
    t1 = load(os.path.join(TABLES, "T1_graph_three_way.csv"))
    sets = ["SPIQA multi-hop (cross-paper)", "HotpotQA bridge"]
    short = {sets[0]: "SPIQA cross-paper", sets[1]: "HotpotQA bridge"}

    acc = {}
    for r in t1:
        qs, system = r["question set"], r["system"]
        if qs not in sets:
            continue
        for key, _, _, _ in MODELS:
            cell = r.get(key, "\u2014")
            if cell and not cell.startswith("\u2014") and cell != "—":
                acc.setdefault(qs, {}).setdefault(system, {})[key] = float(
                    cell.split()[0]
                )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, qs in zip(axes, sets):
        x = range(len(MODELS))
        d_kg = [acc[qs]["+KG"][k] - acc[qs]["baseline"][k] for k, _, _, _ in MODELS]
        d_kgret = [
            acc[qs]["+KGret"][k] - acc[qs]["baseline"][k] for k, _, _, _ in MODELS
        ]
        ax.axhline(0, color="0.55", lw=0.8, zorder=1)
        ax.bar(
            [i - 0.18 for i in x],
            d_kg,
            width=0.34,
            zorder=3,
            color=KG_COLOUR,
            edgecolor="black",
            linewidth=0.6,
            label="+KG (generation stage)",
        )
        ax.bar(
            [i + 0.18 for i in x],
            d_kgret,
            width=0.34,
            zorder=3,
            color=KGRET_COLOUR,
            edgecolor="black",
            linewidth=0.6,
            label="+KGret (retrieval stage)",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [
                lbl.replace(" 3.1 Flash-Lite", "\n3.1 F-L").replace(
                    "Llama 4 ", "Llama 4\n"
                )
                for _, lbl, _, _ in MODELS
            ],
            fontsize=6.5,
        )
        ax.set_title(short[qs])
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5, zorder=0)
    axes[0].set_ylabel("$\\Delta$ accuracy vs baseline")
    axes[0].legend(frameon=False, loc="upper left")
    out = os.path.join(FIGS, "fig4_stage.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig5_leakage():
    """Baseline accuracy conditioned on evidence completeness, per corpus.
    PubLayNet's 0.000 incomplete-evidence floor against 0.35-0.71 on the
    prominent corpora is the parametric-leakage contrast."""
    t3 = load(os.path.join(TABLES, "T3_evidence_conditioned.csv"))
    rows = [
        r
        for r in t3
        if r["system"] == "baseline" and "within-paper" not in r["question set"]
    ]
    labels = {
        "PubLayNet multi-hop (within-page)": "PubLayNet\nmulti-hop",
        "SPIQA multi-hop (cross-paper)": "SPIQA\ncross-paper",
        "HotpotQA bridge": "HotpotQA\nbridge",
        "HotpotQA comparison (control)": "HotpotQA\ncomparison",
    }
    names, comp, incomp = [], [], []
    for r in rows:
        names.append(labels.get(r["question set"], r["question set"]))
        comp.append(float(r["acc | complete"]))
        v = r["acc | incomplete"]
        incomp.append(float(v) if v not in ("\u2014", "—", "") else 0.0)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = range(len(names))
    ax.bar(
        [i - 0.19 for i in x],
        comp,
        width=0.36,
        zorder=3,
        color=KGRET_COLOUR,
        edgecolor="black",
        linewidth=0.6,
        label="evidence complete",
    )
    ax.bar(
        [i + 0.19 for i in x],
        incomp,
        width=0.36,
        zorder=3,
        color=KG_COLOUR,
        edgecolor="black",
        linewidth=0.6,
        label="evidence incomplete",
    )
    for i, v in enumerate(incomp):
        ax.text(i + 0.19, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5, zorder=4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("Baseline accuracy")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5, zorder=0)
    ax.legend(frameon=False, loc="upper left")
    out = os.path.join(FIGS, "fig5_leakage.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
