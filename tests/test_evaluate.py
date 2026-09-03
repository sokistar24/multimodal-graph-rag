import csv
import json

import pytest
from conftest import FakeModelClient

from multimodal_graph_rag.errors import ConfigurationError, ProviderError
from multimodal_graph_rag.pipelines.evaluate import EvaluationSettings, run_evaluation
from multimodal_graph_rag.retrieval.graph import default_cache_path

pytest.importorskip("faiss")


def _settings(tmp_path, corpus_dir, question_file, triple_cache, systems, **overrides):
    overrides.setdefault("model", "gpt4o-mini")
    return EvaluationSettings(
        question_file=question_file,
        systems=systems,
        corpus_dir=corpus_dir,
        pricing_captured_on="2026-01-01",
        cache_dir=tmp_path / ".cache",
        graph_cache_dir=triple_cache.parent,
        results_dir=tmp_path / "runs",
        **overrides,
    )


def test_graph_cache_path_matches_released_layout(corpus_dir, triple_cache):
    assert default_cache_path(corpus_dir, triple_cache.parent) == triple_cache


def test_full_run_writes_summary_detail_and_trace(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    settings = _settings(
        tmp_path,
        corpus_dir,
        question_file,
        triple_cache,
        (
            "baseline",
            "+KG",
            "+KGret",
            "control:closed-book",
            "control:shuffled",
            "control:oracle",
            "control:partial-gold",
        ),
    )
    outputs = run_evaluation(settings, fake_client)
    with outputs.summary_path.open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    systems = [row["system"] for row in summary]
    assert systems == [
        "baseline",
        "+KG",
        "+KGret",
        "control:closed-book",
        "control:shuffled",
        "control:oracle",
        "control:partial-gold",
    ]
    by_system = {row["system"]: row for row in summary}
    assert all(row["acc"] == "1.0" for row in summary)
    assert by_system["baseline"]["n_errors"] == "0"
    assert by_system["control:oracle"]["complete"] == "1.0"
    assert by_system["control:closed-book"]["recall"] == "0.0"
    assert float(by_system["+KGret"]["complete"]) >= float(
        by_system["baseline"]["complete"]
    )
    assert by_system["baseline"]["config_hash"] == outputs.config_hash
    assert by_system["baseline"]["pricing_captured_on"] == "2026-01-01"
    assert float(by_system["baseline"]["judge_cost_usd"]) > 0

    with outputs.detail_path.open(newline="", encoding="utf-8") as stream:
        detail = list(csv.DictReader(stream))
    assert len(detail) == 3
    assert detail[1]["expected_source"] == "alpha.txt|beta.txt"
    assert "question_id" in detail[0]

    traces = [
        json.loads(line)
        for line in outputs.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(traces) == 3 * 7
    kgret = [
        t
        for t in traces
        if t["system"] == "+KGret" and t["question_id"] == detail[1]["question_id"]
    ][0]
    reasons = {item["selection_reason"] for item in kgret["evidence"]["items"]}
    assert "dense" in reasons
    assert kgret["evidence"]["candidate_budget"] == 5


def test_provider_failure_aborts_and_keeps_partial_rows(
    tmp_path, corpus_dir, question_file, triple_cache
):
    client = FakeModelClient(fail_after_calls=6)
    settings = _settings(
        tmp_path, corpus_dir, question_file, triple_cache, ("baseline",)
    )
    with pytest.raises(ProviderError):
        run_evaluation(settings, client)
    partial = list((tmp_path / "runs").glob("detail_partial_*.csv"))
    assert len(partial) == 1
    assert not list((tmp_path / "runs").glob("summary_*.csv"))


def test_mismatched_corpus_is_refused(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    other = tmp_path / "other"
    other.mkdir()
    (other / "zzz.txt").write_text("unrelated text about nothing", encoding="utf-8")
    settings = _settings(tmp_path, other, question_file, triple_cache, ("baseline",))
    with pytest.raises(ConfigurationError, match="do not match"):
        run_evaluation(settings, fake_client)


def test_settings_validate_generator_and_systems(
    tmp_path, corpus_dir, question_file, triple_cache
):
    with pytest.raises(ConfigurationError):
        _settings(
            tmp_path,
            corpus_dir,
            question_file,
            triple_cache,
            ("baseline",),
            model="deepseek",
        )
    with pytest.raises(ConfigurationError, match="unknown system"):
        _settings(tmp_path, corpus_dir, question_file, triple_cache, ("nope",))
