"""Reported tables generated from one run selection.

T1  graph three-way comparison (baseline, +KG, +KGret) per generator
T2  bridge versus comparison mechanism, means over generators
T3  accuracy conditioned on evidence completeness
T4  figure-question integrity tiers
T5  every selected cell, flat
T6  efficiency
T7  evidence-control arms where present
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from ..clients import DIAGNOSTICS, GENERATORS
from ..evaluation.integrity import audit_answer_recoverability
from ..schemas import load_questions
from .collect import RunSelection

MODELS: tuple[str, ...] = GENERATORS
SYSTEMS = ("baseline", "+KG", "+KGret", "+multimodal", "+both")
CONTROL_SYSTEMS = (
    "control:closed-book",
    "control:shuffled",
    "control:oracle",
    "control:partial-gold",
)
GRAPH_SETS = (
    "publaynet_multihop",
    "spiqa_multihop",
    "spiqa_multihop_cross",
    "hotpotqa_bridge",
    "hotpotqa_comparison",
)
SET_LABEL = {
    "publaynet_multihop": "PubLayNet multi-hop (within-page)",
    "spiqa_multihop": "SPIQA multi-hop (within-paper)",
    "spiqa_multihop_cross": "SPIQA multi-hop (cross-paper, graph-seeded)",
    "hotpotqa_bridge": "HotpotQA bridge",
    "hotpotqa_comparison": "HotpotQA comparison (control)",
}
# Crops whose answers were found verbatim in the pipeline caption during the
# integrity audit; excluded from every headline claim.
CAPTION_CONTAMINATED_SPIQA_CROPS = frozenset(
    {
        "1809.00263v5-Figure12-1.png",
        "1906.10843v1-Figure5-1.png",
        "1705.02946v3-Figure6-1.png",
        "1803.01128v3-Table2-1.png",
        "1802.07351v2-Figure10-1.png",
    }
)
DASH = "—"


def fmt(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_table(
    out_dir: Path,
    name: str,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
    note: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as stream:
        stream.write("| " + " | ".join(header) + " |\n")
        stream.write("|" + "|".join("---" for _ in header) + "|\n")
        stream.writelines(
            "| " + " | ".join(str(x) for x in row) + " |\n" for row in rows
        )
        if note:
            stream.write(f"\n_{note}_\n")
    return csv_path


def table_graph_three_way(selection: RunSelection, out_dir: Path) -> Path:
    header = ["question set", "system", "AllGoldFound", *MODELS]
    rows = []
    for qset in GRAPH_SETS:
        for system in ("baseline", "+KG", "+KGret"):
            values, completeness = [], None
            for model in MODELS:
                cell = selection.get(qset, model, system)
                if cell:
                    values.append(
                        f"{fmt(cell.summary['acc'])} ({fmt(cell.summary['faith'])})"
                    )
                    completeness = cell.summary["complete"]
                else:
                    values.append(DASH)
            if any(v != DASH for v in values):
                rows.append([SET_LABEL[qset], system, completeness or DASH, *values])
    return write_table(
        out_dir,
        "T1_graph_three_way",
        header,
        rows,
        "Cells: accuracy (faithfulness). AllGoldFound is retrieval completeness "
        "over all gold sources; identical across generators within a system "
        "because retrieval is fixed. +KGret expands the candidate set, so its "
        "completeness differs by design.",
    )


def table_bridge_vs_comparison(selection: RunSelection, out_dir: Path) -> Path:
    header = [
        "set",
        "system",
        "AllGoldFound",
        "mean acc",
        "mean faith",
        "delta acc vs baseline",
    ]
    rows = []
    for qset in ("hotpotqa_bridge", "hotpotqa_comparison"):
        base_acc = None
        for system in ("baseline", "+KG", "+KGret"):
            accs, faiths, completeness = [], [], None
            for model in MODELS:
                cell = selection.get(qset, model, system)
                if cell:
                    accs.append(float(cell.summary["acc"]))
                    faiths.append(float(cell.summary["faith"]))
                    completeness = cell.summary["complete"]
            if not accs:
                continue
            mean_acc = sum(accs) / len(accs)
            if system == "baseline":
                base_acc = mean_acc
            rows.append(
                [
                    SET_LABEL[qset],
                    system,
                    completeness,
                    fmt(mean_acc),
                    fmt(sum(faiths) / len(faiths)),
                    f"{mean_acc - base_acc:+.3f}" if base_acc is not None else DASH,
                ]
            )
    return write_table(
        out_dir,
        "T2_bridge_vs_comparison",
        header,
        rows,
        "Bridge questions have low baseline completeness (the need); comparison "
        "questions retrieve fully (the control). Means over generators are "
        "descriptive only: generators are not independent replications.",
    )


def table_evidence_conditioned(selection: RunSelection, out_dir: Path) -> Path:
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
        for system in ("baseline", "+KG", "+KGret"):
            n_complete = n_incomplete = acc_complete = acc_incomplete = 0
            for model in MODELS:
                cell = selection.get(qset, model, system)
                if not cell or cell.detail_path is None:
                    continue
                for row in cell.detail_rows():
                    acc_col, comp_col = f"{system}_acc", f"{system}_complete"
                    if row.get(acc_col) in ("", None):
                        continue
                    complete = float(row.get(comp_col, 0) or 0) >= 1.0
                    acc = int(float(row[acc_col]))
                    if complete:
                        n_complete += 1
                        acc_complete += acc
                    else:
                        n_incomplete += 1
                        acc_incomplete += acc
            if n_complete + n_incomplete:
                rows.append(
                    [
                        SET_LABEL[qset],
                        system,
                        fmt(acc_complete / n_complete) if n_complete else DASH,
                        fmt(acc_incomplete / n_incomplete) if n_incomplete else DASH,
                        n_complete,
                        n_incomplete,
                    ]
                )
    return write_table(
        out_dir,
        "T3_evidence_conditioned",
        header,
        rows,
        "Pooled over generators. 'acc | incomplete' is accuracy without complete "
        "gold provenance; its source (non-gold support, partial evidence, or "
        "closed-book knowledge) is established by the control arms and human "
        "audit, not assumed to be parametric memory.",
    )


def _spiqa_tiers(question_file: Path | None, corpus_dir: Path | None) -> dict[str, str]:
    if question_file is None or not question_file.is_file():
        return {}
    tiers: dict[str, str] = {}
    for question in load_questions(question_file):
        source = question.gold_sources[0]
        if source in CAPTION_CONTAMINATED_SPIQA_CROPS:
            tiers[source] = "caption-contaminated"
            continue
        if corpus_dir is None or not corpus_dir.is_dir():
            tiers[source] = "clean (corpus absent: tier unknown)"
            continue
        paper = corpus_dir / f"{source.split('-')[0]}.txt"
        text = (
            paper.read_text(encoding="utf-8", errors="replace")
            if paper.is_file()
            else ""
        )
        audit = audit_answer_recoverability(question.answer, text)
        tiers[source] = (
            "text-recoverable"
            if audit.exact_match
            else "exact-match-screened pixel-only"
        )
    return tiers


def table_figure_integrity(
    selection: RunSelection,
    out_dir: Path,
    *,
    spiqa_figure_questions: Path | None,
    spiqa_corpus: Path | None,
) -> Path:
    header = ["set / tier", "n", "model", "text-only baseline acc", "+multimodal acc"]
    rows = []
    for qset in ("publaynet_figures", "publaynet_figures_caption"):
        for model in MODELS:
            base = selection.get(qset, model, "baseline")
            multimodal = selection.get(qset, model, "+multimodal")
            if base:
                rows.append(
                    [
                        qset,
                        base.summary["n_questions"],
                        model,
                        fmt(base.summary["acc"]),
                        fmt(multimodal.summary["acc"]) if multimodal else DASH,
                    ]
                )
    tiers = _spiqa_tiers(spiqa_figure_questions, spiqa_corpus)
    for model in MODELS:
        base = selection.get("spiqa_figures", model, "baseline")
        multimodal = selection.get("spiqa_figures", model, "+multimodal")
        if not base or base.detail_path is None:
            continue
        details = {row["expected_source"]: row for row in base.detail_rows()}
        has_multimodal = bool(multimodal) and any(
            "+multimodal_acc" in row for row in details.values()
        )
        for tier in sorted(set(tiers.values())):
            sources = [s for s, t in tiers.items() if t == tier and s in details]
            if not sources:
                continue
            base_hits = sum(int(float(details[s]["baseline_acc"])) for s in sources)
            multimodal_hits = (
                sum(int(float(details[s]["+multimodal_acc"])) for s in sources)
                if has_multimodal
                else None
            )
            rows.append(
                [
                    f"spiqa_figures [{tier}]",
                    len(sources),
                    model,
                    f"{base_hits}/{len(sources)}",
                    f"{multimodal_hits}/{len(sources)}"
                    if multimodal_hits is not None
                    else DASH,
                ]
            )
    return write_table(
        out_dir,
        "T4_figure_integrity",
        header,
        rows,
        "SPIQA figures tiered by leak path: caption-contaminated (excluded from "
        "headline claims), text-recoverable (answer string present in corpus "
        "prose), and exact-match-screened pixel-only. Screening is normalised "
        "string matching; it is not a verified pixel-only guarantee.",
    )


def table_all_runs(selection: RunSelection, out_dir: Path) -> Path:
    header = [
        "question set",
        "model",
        "system",
        "recall",
        "complete",
        "acc",
        "faith",
        "rel",
        "run_id",
    ]
    rows = []
    for (qset, model, system), cell in sorted(selection.cells.items()):
        if model in DIAGNOSTICS:
            continue
        summary = cell.summary
        rows.append(
            [
                qset,
                model,
                system,
                summary["recall"],
                summary["complete"],
                summary["acc"],
                summary["faith"],
                summary["rel"],
                cell.run_id,
            ]
        )
    return write_table(
        out_dir,
        "T5_all_runs_flat",
        header,
        rows,
        f"Every selected cell with its run id. Selection: {selection.provenance}.",
    )


def table_efficiency(selection: RunSelection, out_dir: Path) -> Path:
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
    for qset in ("publaynet_figures", "spiqa_multihop_cross", "hotpotqa_bridge"):
        for model in MODELS:
            for system in SYSTEMS:
                cell = selection.get(qset, model, system)
                if cell:
                    summary = cell.summary
                    rows.append(
                        [
                            qset,
                            model,
                            system,
                            summary["mean_in_tok"],
                            summary["mean_out_tok"],
                            summary["mean_latency_ms"],
                            summary["cost_per_100q_usd"],
                        ]
                    )
    return write_table(out_dir, "T6_efficiency", header, rows)


def table_controls(selection: RunSelection, out_dir: Path) -> Path | None:
    header = ["question set", "model", "system", "acc", "faith", "complete"]
    rows = []
    for (qset, model, system), cell in sorted(selection.cells.items()):
        if system in CONTROL_SYSTEMS or system == "baseline":
            summary = cell.summary
            rows.append(
                [
                    qset,
                    model,
                    system,
                    summary["acc"],
                    summary["faith"],
                    summary["complete"],
                ]
            )
    if not any(row[2] in CONTROL_SYSTEMS for row in rows):
        return None
    return write_table(
        out_dir,
        "T7_evidence_controls",
        header,
        rows,
        "Closed-book, shuffled, oracle, and partial-gold arms next to the normal "
        "baseline for the same generator and question set.",
    )


def build_tables(
    selection: RunSelection,
    out_dir: str | Path,
    *,
    spiqa_figure_questions: str | Path | None = None,
    spiqa_corpus: str | Path | None = None,
) -> list[Path]:
    destination = Path(out_dir)
    written = [
        table_graph_three_way(selection, destination),
        table_bridge_vs_comparison(selection, destination),
        table_evidence_conditioned(selection, destination),
        table_figure_integrity(
            selection,
            destination,
            spiqa_figure_questions=Path(spiqa_figure_questions)
            if spiqa_figure_questions
            else None,
            spiqa_corpus=Path(spiqa_corpus) if spiqa_corpus else None,
        ),
        table_all_runs(selection, destination),
        table_efficiency(selection, destination),
    ]
    controls = table_controls(selection, destination)
    if controls:
        written.append(controls)
    return written
