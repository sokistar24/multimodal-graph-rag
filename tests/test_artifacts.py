import csv
import json

import pytest

from multimodal_graph_rag.artifacts import (
    build_figures,
    build_tables,
    file_sha256,
    freeze_manifest,
    load_frozen_summary_rows,
    scan_runs,
    select_from_manifest,
    validate_manifest,
)
from multimodal_graph_rag.errors import ConfigurationError

SUMMARY_FIELDS = [
    "run_id",
    "question_set",
    "question_type",
    "model",
    "vendor",
    "access",
    "n_questions",
    "system",
    "config_hash",
    "recall",
    "complete",
    "mrr",
    "acc",
    "faith",
    "rel",
    "mean_in_tok",
    "mean_out_tok",
    "mean_latency_ms",
    "total_cost_usd",
    "cost_per_100q_usd",
    "n_errors",
]


def _write_run(directory, qset, model, stamp, systems, *, recall="0.8"):
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"run-{stamp}"
    summary = directory / f"summary_questions_{qset}_{model}_{stamp}.csv"
    detail = directory / f"detail_questions_{qset}_{model}_{stamp}.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for system, acc in systems.items():
            writer.writerow(
                {
                    "run_id": run_id,
                    "question_set": f"questions_{qset}",
                    "question_type": "text",
                    "model": model,
                    "vendor": "v",
                    "access": "closed",
                    "n_questions": 2,
                    "system": system,
                    "config_hash": "h",
                    "recall": recall,
                    "complete": "0.5",
                    "mrr": "1.0",
                    "acc": acc,
                    "faith": "1.0",
                    "rel": "1.0",
                    "mean_in_tok": "500",
                    "mean_out_tok": "20",
                    "mean_latency_ms": "100",
                    "total_cost_usd": "0.001",
                    "cost_per_100q_usd": "0.05",
                    "n_errors": "0",
                }
            )
    fields = ["question_id", "expected_source"] + [
        f"{s}_{m}" for s in systems for m in ("acc", "complete", "mrr")
    ]
    with detail.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(2):
            row = {"question_id": f"q{index}", "expected_source": "a.txt"}
            for system in systems:
                row[f"{system}_acc"] = 1 if index == 0 else 0
                row[f"{system}_complete"] = 1 if index == 0 else 0
                row[f"{system}_mrr"] = 1.0 if index == 0 else 0.0
            writer.writerow(row)
    return summary, detail


def test_manifest_locks_exact_run_and_checksum(tmp_path):
    summary, detail = _write_run(
        tmp_path / "runs",
        "hotpotqa_bridge",
        "gpt4o-mini",
        "20260101_000000",
        {"baseline": "0.5"},
    )
    manifest = freeze_manifest([summary], root=tmp_path, release_id="r1", commit="abc")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_manifest(path, tmp_path) == []
    assert (
        load_frozen_summary_rows(path, tmp_path)[0]["_manifest_run_id"]
        == "run-20260101_000000"
    )
    detail.write_text("tampered", encoding="utf-8")
    assert any(
        "checksum mismatch" in error for error in validate_manifest(path, tmp_path)
    )


def test_manifest_refuses_archive_paths(tmp_path):
    summary, _ = _write_run(
        tmp_path / "archive" / "runs",
        "hotpotqa_bridge",
        "gpt4o-mini",
        "20260101_000000",
        {"baseline": "0.5"},
    )
    manifest = freeze_manifest([summary], root=tmp_path, release_id="r1", commit="abc")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any("archive" in error for error in validate_manifest(path, tmp_path))


def test_scan_takes_newest_valid_run_without_backfilling(tmp_path):
    runs = tmp_path / "runs"
    _write_run(
        runs,
        "hotpotqa_bridge",
        "gpt4o-mini",
        "20260101_000000",
        {"baseline": "0.5", "+KG": "0.6", "+KGret": "0.7"},
    )
    _write_run(
        runs, "hotpotqa_bridge", "gpt4o-mini", "20260102_000000", {"baseline": "0.4"}
    )
    _write_run(
        runs,
        "hotpotqa_comparison",
        "gpt4o-mini",
        "20260103_000000",
        {"baseline": "0.9"},
        recall="0.0",
    )
    selection = scan_runs(runs)
    assert (
        selection.get("hotpotqa_bridge", "gpt4o-mini", "baseline").summary["acc"]
        == "0.4"
    )
    assert selection.get("hotpotqa_bridge", "gpt4o-mini", "+KG") is None
    assert selection.get("hotpotqa_comparison", "gpt4o-mini", "baseline") is None


def test_tables_and_figures_build_from_a_selection(tmp_path):
    runs = tmp_path / "runs"
    for model in ("gpt4o-mini", "llama4-scout"):
        _write_run(
            runs,
            "hotpotqa_bridge",
            model,
            "20260101_000000",
            {
                "baseline": "0.5",
                "+KG": "0.5",
                "+KGret": "0.8",
                "control:closed-book": "0.2",
            },
        )
        _write_run(
            runs,
            "spiqa_multihop_cross",
            model,
            "20260101_000000",
            {"baseline": "0.5", "+KG": "0.4", "+KGret": "0.7"},
        )
        _write_run(
            runs,
            "publaynet_figures",
            model,
            "20260101_000000",
            {"baseline": "0.1", "+multimodal": "0.3"},
        )
    selection = scan_runs(runs)
    tables = build_tables(selection, tmp_path / "tables")
    names = {path.name for path in tables}
    assert {
        "T1_graph_three_way.csv",
        "T3_evidence_conditioned.csv",
        "T5_all_runs_flat.csv",
        "T7_evidence_controls.csv",
    } <= names
    with (tmp_path / "tables" / "T3_evidence_conditioned.csv").open(
        newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    bridge = [
        r
        for r in rows
        if r["system"] == "baseline" and "HotpotQA bridge" == r["question set"]
    ][0]
    assert bridge["n complete"] == "2" and bridge["n incomplete"] == "2"
    figures = build_figures(selection, tmp_path / "tables", tmp_path / "figures")
    assert {path.name for path in figures} == {
        "fig2_cost_accuracy.pdf",
        "fig3_image_tokens.pdf",
        "fig4_stage.pdf",
        "fig5_leakage.pdf",
    }


def test_manifest_selection_rejects_duplicate_cells(tmp_path):
    runs = tmp_path / "runs"
    first, _ = _write_run(
        runs, "hotpotqa_bridge", "gpt4o-mini", "20260101_000000", {"baseline": "0.5"}
    )
    second, _ = _write_run(
        runs, "hotpotqa_bridge", "gpt4o-mini", "20260102_000000", {"baseline": "0.6"}
    )
    manifest = freeze_manifest(
        [first, second], root=tmp_path, release_id="r", commit="c"
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="two runs"):
        select_from_manifest(path, tmp_path)


def test_file_sha256_is_stable(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"abc")
    assert (
        file_sha256(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
