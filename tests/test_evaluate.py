import csv
import json

import pytest
from conftest import FakeModelClient

from multimodal_graph_rag.config import ExperimentConfig
from multimodal_graph_rag.errors import (
    BudgetExceededError,
    ConfigurationError,
    ProviderError,
)
from multimodal_graph_rag.pipelines.evaluate import EvaluationSettings, run_evaluation
from multimodal_graph_rag.retrieval.comparators import write_retrievals
from multimodal_graph_rag.retrieval.graph import default_cache_path
from multimodal_graph_rag.schemas import load_questions

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


def test_relevancy_is_no_longer_judged(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    settings = _settings(
        tmp_path, corpus_dir, question_file, triple_cache, ("baseline",)
    )
    outputs = run_evaluation(settings, fake_client)
    relevancy_calls = [user for _, user, _ in fake_client.calls if "RELEVANT" in user]
    assert relevancy_calls == []
    with outputs.summary_path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert "rel" not in header
    trace = json.loads(outputs.trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert set(trace["raw_judges"]) == {"accuracy", "faithfulness"}


def test_budget_cap_aborts_the_run_and_keeps_partial_rows(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    settings = _settings(
        tmp_path,
        corpus_dir,
        question_file,
        triple_cache,
        ("baseline",),
        budget_usd=1e-9,
    )
    with pytest.raises(BudgetExceededError, match="budget"):
        run_evaluation(settings, fake_client)
    assert len(list((tmp_path / "runs").glob("detail_partial_*.csv"))) == 1
    assert not list((tmp_path / "runs").glob("summary_*.csv"))


def test_budget_must_be_positive_when_set(
    tmp_path, corpus_dir, question_file, triple_cache
):
    with pytest.raises(ConfigurationError, match="budget_usd"):
        _settings(
            tmp_path,
            corpus_dir,
            question_file,
            triple_cache,
            ("baseline",),
            budget_usd=0.0,
        )


def test_config_budget_reaches_settings_and_identity(tmp_path):
    base = dict(
        name="cap",
        corpus="hotpotqa",
        corpus_dir="hotpotqa_corpus",
        question_set="data/questions/questions_hotpotqa_bridge.json",
        systems=("baseline",),
        generators=("gpt4o-mini",),
        pricing_snapshot="configs/pricing_2026-09-10.json",
    )
    capped = ExperimentConfig(**base, budget_usd=2.5)
    settings = EvaluationSettings.from_config(
        capped, "gpt4o-mini", pricing_captured_on="x"
    )
    assert settings.budget_usd == 2.5
    assert capped.config_hash != ExperimentConfig(**base).config_hash


def _comparator_file(tmp_path, corpus_dir, question_file):
    questions = load_questions(question_file)
    path = tmp_path / "hipporag_retrievals.json"
    write_retrievals(
        path,
        retrievals={
            q.id: [
                {"source": s, "text": f"chunk of {s}", "score": 1.0}
                for s in q.gold_sources
            ]
            for q in questions
        },
        corpus_dir=corpus_dir,
        question_file=question_file,
        comparator="hipporag2",
        num_to_retrieve=5,
    )
    return path


def test_hipporag_arm_serves_precomputed_retrievals(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    comparator = _comparator_file(tmp_path, corpus_dir, question_file)
    settings = _settings(
        tmp_path,
        corpus_dir,
        question_file,
        triple_cache,
        ("baseline", "+HippoRAG"),
        comparator_file=comparator,
    )
    outputs = run_evaluation(settings, fake_client)
    with outputs.summary_path.open(newline="", encoding="utf-8") as stream:
        by_system = {row["system"]: row for row in csv.DictReader(stream)}
    assert by_system["+HippoRAG"]["complete"] == "1.0"
    traces = [
        json.loads(line)
        for line in outputs.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    hippo = [t for t in traces if t["system"] == "+HippoRAG"][0]
    assert {item["selection_reason"] for item in hippo["evidence"]["items"]} == {
        "hipporag"
    }


def test_hipporag_arm_without_retrievals_is_a_configuration_error(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    settings = _settings(
        tmp_path, corpus_dir, question_file, triple_cache, ("+HippoRAG",)
    )
    with pytest.raises(ConfigurationError, match="comparator"):
        run_evaluation(settings, fake_client)


def test_config_comparator_path_reaches_settings(tmp_path):
    config = ExperimentConfig(
        name="cmp",
        corpus="hotpotqa",
        corpus_dir="hotpotqa_corpus",
        question_set="data/questions/questions_hotpotqa_bridge.json",
        systems=("baseline", "+HippoRAG"),
        generators=("gpt4o-mini",),
        pricing_snapshot="configs/pricing_2026-09-10.json",
        comparator_retrievals=".cache/hipporag/hotpotqa/retrievals.json",
    )
    settings = EvaluationSettings.from_config(
        config, "gpt4o-mini", pricing_captured_on="x", root=tmp_path
    )
    assert settings.comparator_file == tmp_path / config.comparator_retrievals
