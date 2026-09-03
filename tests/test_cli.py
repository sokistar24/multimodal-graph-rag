import json

from multimodal_graph_rag.cli import build_parser, main


def test_dry_run_evaluate_reports_every_generator(capsys):
    assert (
        main(["evaluate", "--config", "configs/primary_revision.json", "--dry-run"])
        == 0
    )
    out = capsys.readouterr().out
    for model in ("gpt4o-mini", "gemini-flash-lite", "llama4-scout", "llama4-maverick"):
        assert model in out
    assert "control:closed-book" in out


def test_invalid_manifest_is_reported_not_raised(tmp_path, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [{"run_id": "x", "summary": "nope.csv", "detail": "nope.csv"}],
            }
        )
    )
    assert main(["validate-release", "--manifest", str(manifest)]) == 1
    assert "missing summary" in capsys.readouterr().out


def test_package_errors_exit_with_code_two(tmp_path, capsys):
    assert main(["evaluate", "--config", str(tmp_path / "missing.json")]) == 2
    assert "error:" in capsys.readouterr().err


def test_parser_lists_all_commands():
    commands = build_parser()._subparsers._group_actions[0].choices
    assert {
        "ingest",
        "questions",
        "evaluate",
        "retrieval-ab",
        "artifacts",
        "validate-release",
        "freeze-manifest",
        "audit-graph",
        "audit-figures",
        "paired-analysis",
        "clean-questions",
    } <= set(commands)
