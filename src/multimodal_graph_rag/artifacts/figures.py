"""Reported figures rendered from a run selection and the generated tables.

    fig2_cost_accuracy.pdf   cost versus accuracy on pixel-only figure questions
    fig3_image_tokens.pdf    input tokens per question on +multimodal
    fig4_stage.pdf           +KG versus +KGret accuracy deltas per generator
    fig5_leakage.pdf         baseline accuracy conditioned on evidence completeness

Only the production generators are plotted; the diagnostic GPT-4o run is not a
fifth system. Figure 2's ceiling is the operational Recall@1 of image
retrieval (one crop is supplied to the generator), computed from the detail
rows, not the diagnostic Recall@k over scored candidates.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before import)

from .collect import RunSelection  # noqa: E402

PIXEL_ONLY_SET = "publaynet_figures"
BASELINE_COLOUR = "#1f77b4"
KG_COLOUR = "#c8553d"
KGRET_COLOUR = "#2e6f95"
MODELS = (
    ("llama4-scout", "Llama 4 Scout", "#1b9e77", "o"),
    ("llama4-maverick", "Llama 4 Maverick", "#66a61e", "o"),
    ("gemini-flash-lite", "Gemini 3.1 Flash-Lite", "#d95f02", "s"),
    ("gpt4o-mini", "GPT-4o-mini", "#e7298a", "s"),
)
STYLE = {
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


def _short_label(label: str) -> str:
    return label.replace(" 3.1 Flash-Lite", "\n3.1 Flash-Lite").replace(
        "Llama 4 ", "Llama 4\n"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def recall_at_one(selection: RunSelection, qset: str, system: str) -> float | None:
    """Fraction of questions whose single gold image was ranked first."""
    hits = total = 0
    for key, _, _, _ in MODELS:
        cell = selection.get(qset, key, system)
        if cell is None or cell.detail_path is None:
            continue
        for row in cell.detail_rows():
            value = row.get(f"{system}_mrr")
            if value in (None, ""):
                continue
            total += 1
            hits += float(value) >= 1.0
        break
    return hits / total if total else None


def figure_cost_accuracy(selection: RunSelection, out_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    plotted = False
    for key, label, colour, marker in MODELS:
        cell = selection.get(PIXEL_ONLY_SET, key, "+multimodal")
        if cell is None:
            continue
        x = float(cell.summary["cost_per_100q_usd"])
        y = float(cell.summary["acc"])
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
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ceiling = recall_at_one(selection, PIXEL_ONLY_SET, "+multimodal")
    if ceiling is not None:
        ax.axhline(ceiling, ls="--", lw=0.9, c="#555555", zorder=1)
        ax.annotate(
            f"operational retrieval ceiling (Recall@1 = {ceiling:.3f})",
            (0.0042, ceiling + 0.012),
            fontsize=6.5,
            color="#444444",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 100 questions (USD, log scale)")
    ax.set_ylabel("Answer accuracy")
    ax.set_ylim(0, (ceiling or 0.4) + 0.20)
    ax.set_xlim(0.004, 0.30)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5, zorder=0)
    ax.legend(
        loc="upper right",
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.3,
        labelspacing=0.35,
    )
    path = out_dir / "fig2_cost_accuracy.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_image_tokens(selection: RunSelection, out_dir: Path) -> Path | None:
    labels, values, colours = [], [], []
    for key, label, colour, _ in MODELS:
        cell = selection.get(PIXEL_ONLY_SET, key, "+multimodal")
        if cell is None:
            continue
        labels.append(_short_label(label))
        values.append(float(cell.summary["mean_in_tok"]))
        colours.append(colour)
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.bar(
        range(len(values)),
        values,
        color=colours,
        edgecolor="black",
        linewidth=0.7,
        width=0.6,
        zorder=3,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.07,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            zorder=4,
        )
    baseline = [
        float(selection.get(PIXEL_ONLY_SET, key, "baseline").summary["mean_in_tok"])
        for key, _, _, _ in MODELS
        if selection.get(PIXEL_ONLY_SET, key, "baseline")
    ]
    if baseline:
        low, high, mean = min(baseline), max(baseline), sum(baseline) / len(baseline)
        ax.axhline(mean, ls="--", lw=1.3, c=BASELINE_COLOUR, zorder=2)
        ax.annotate(
            f"text-only baseline\n({low:.0f}–{high:.0f} tok, all models)",
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
    path = out_dir / "fig3_image_tokens.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_stage(tables_dir: Path, out_dir: Path) -> Path | None:
    table = tables_dir / "T1_graph_three_way.csv"
    if not table.is_file():
        return None
    sets = ["SPIQA multi-hop (cross-paper, graph-seeded)", "HotpotQA bridge"]
    short = {sets[0]: "SPIQA cross-paper (graph-seeded)", sets[1]: "HotpotQA bridge"}
    accuracy: dict[str, dict[str, dict[str, float]]] = {}
    for row in _read_csv(table):
        qset, system = row["question set"], row["system"]
        if qset not in sets:
            continue
        for key, _, _, _ in MODELS:
            cell = row.get(key, "—")
            if cell and not cell.startswith("—"):
                accuracy.setdefault(qset, {}).setdefault(system, {})[key] = float(
                    cell.split()[0]
                )
    if not all(
        qset in accuracy and {"baseline", "+KG", "+KGret"} <= set(accuracy[qset])
        for qset in sets
    ):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, qset in zip(axes, sets, strict=True):
        positions = range(len(MODELS))
        delta_kg = [
            accuracy[qset]["+KG"].get(k, 0.0) - accuracy[qset]["baseline"].get(k, 0.0)
            for k, _, _, _ in MODELS
        ]
        delta_kgret = [
            accuracy[qset]["+KGret"].get(k, 0.0)
            - accuracy[qset]["baseline"].get(k, 0.0)
            for k, _, _, _ in MODELS
        ]
        ax.axhline(0, color="0.55", lw=0.8, zorder=1)
        ax.bar(
            [i - 0.18 for i in positions],
            delta_kg,
            width=0.34,
            zorder=3,
            color=KG_COLOUR,
            edgecolor="black",
            linewidth=0.6,
            label="+KG (generation stage)",
        )
        ax.bar(
            [i + 0.18 for i in positions],
            delta_kgret,
            width=0.34,
            zorder=3,
            color=KGRET_COLOUR,
            edgecolor="black",
            linewidth=0.6,
            label="+KGret (retrieval stage)",
        )
        ax.set_xticks(list(positions))
        ax.set_xticklabels(
            [
                _short_label(label).replace("Flash-Lite", "F-L")
                for _, label, _, _ in MODELS
            ],
            fontsize=6.5,
        )
        ax.set_title(short[qset])
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5, zorder=0)
    axes[0].set_ylabel("$\\Delta$ accuracy vs baseline")
    axes[0].legend(frameon=False, loc="upper left")
    path = out_dir / "fig4_stage.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_leakage(tables_dir: Path, out_dir: Path) -> Path | None:
    table = tables_dir / "T3_evidence_conditioned.csv"
    if not table.is_file():
        return None
    labels = {
        "PubLayNet multi-hop (within-page)": "PubLayNet\nmulti-hop",
        "SPIQA multi-hop (cross-paper, graph-seeded)": "SPIQA\ncross-paper",
        "HotpotQA bridge": "HotpotQA\nbridge",
        "HotpotQA comparison (control)": "HotpotQA\ncomparison",
    }
    rows = [
        r
        for r in _read_csv(table)
        if r["system"] == "baseline" and "within-paper" not in r["question set"]
    ]
    if not rows:
        return None
    names, complete, incomplete = [], [], []
    for row in rows:
        names.append(labels.get(row["question set"], row["question set"]))
        complete.append(
            float(row["acc | complete"]) if row["acc | complete"] != "—" else 0.0
        )
        value = row["acc | incomplete"]
        incomplete.append(float(value) if value != "—" else 0.0)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    positions = range(len(names))
    ax.bar(
        [i - 0.19 for i in positions],
        complete,
        width=0.36,
        zorder=3,
        color=KGRET_COLOUR,
        edgecolor="black",
        linewidth=0.6,
        label="evidence complete",
    )
    ax.bar(
        [i + 0.19 for i in positions],
        incomplete,
        width=0.36,
        zorder=3,
        color=KG_COLOUR,
        edgecolor="black",
        linewidth=0.6,
        label="evidence incomplete",
    )
    for i, value in enumerate(incomplete):
        ax.text(
            i + 0.19, value + 0.02, f"{value:.2f}", ha="center", fontsize=6.5, zorder=4
        )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("Baseline accuracy")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5, zorder=0)
    ax.legend(frameon=False, loc="upper left")
    path = out_dir / "fig5_leakage.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def build_figures(
    selection: RunSelection, tables_dir: str | Path, out_dir: str | Path
) -> list[Path]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    written = [
        figure_cost_accuracy(selection, destination),
        figure_image_tokens(selection, destination),
        figure_stage(Path(tables_dir), destination),
        figure_leakage(Path(tables_dir), destination),
    ]
    return [path for path in written if path is not None]
