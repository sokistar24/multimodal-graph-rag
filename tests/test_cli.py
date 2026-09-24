import json

from multimodal_graph_rag.cli import build_parser, main
from multimodal_graph_rag.retrieval.comparators import write_retrievals
from multimodal_graph_rag.schemas import load_questions


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


def test_comparator_ab_reports_completeness_without_a_client(
    tmp_path, corpus_dir, question_file, capsys
):
    questions = load_questions(question_file)
    retrievals = tmp_path / "r.json"
    write_retrievals(
        retrievals,
        retrievals={
            q.id: [{"source": q.gold_sources[0], "text": "t", "score": 1.0}]
            for q in questions
        },
        corpus_dir=corpus_dir,
        question_file=question_file,
        comparator="hipporag2",
        num_to_retrieve=1,
    )
    output = tmp_path / "ab.json"
    code = main(
        [
            "comparator-ab",
            "--questions",
            str(question_file),
            "--retrievals",
            str(retrievals),
            "--k",
            "1",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert '"complete": 2' in capsys.readouterr().out
    assert output.is_file()
