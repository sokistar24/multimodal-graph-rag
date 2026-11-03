import csv
import json
from dataclasses import asdict, replace

import pytest
from conftest import FakeModelClient

from multimodal_graph_rag.clients import GENERATOR_MAX_TOKENS
from multimodal_graph_rag.config import ExperimentConfig
from multimodal_graph_rag.errors import (
    BudgetExceededError,
    ConfigurationError,
    MissingEvidenceError,
    ProviderError,
)
from multimodal_graph_rag.evaluation.posthoc import judgements_path
from multimodal_graph_rag.pipelines import evaluate
from multimodal_graph_rag.pipelines.evaluate import (
    LEGACY_GENERATOR_MAX_TOKENS,
    EvaluationSettings,
    run_evaluation,
)
from multimodal_graph_rag.pipelines.systems import (
    PASSAGE_RETRIEVERS,
    RunContext,
    passage_arm_report,
    replay,
    run_bm25,
    run_lexical,
    run_oracle_image,
)
from multimodal_graph_rag.retrieval.baselines import chunk_lexicon
from multimodal_graph_rag.retrieval.comparators import write_retrievals
from multimodal_graph_rag.retrieval.graph import default_cache_path, load_graph
from multimodal_graph_rag.retrieval.text import TextIndex
from multimodal_graph_rag.schemas import content_hash, load_questions

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


def test_a_stopped_run_resumes_to_the_same_result(
    tmp_path, corpus_dir, question_file, triple_cache
):
    systems = ("baseline", "+KG")
    # Six calls per question; the eleventh fails after question 2 has a
    # baseline record but no +KG record, so that question is partial.
    stopped = _settings(tmp_path, corpus_dir, question_file, triple_cache, systems)
    with pytest.raises(ProviderError):
        run_evaluation(stopped, FakeModelClient(fail_after_calls=10))
    trace = next((tmp_path / "runs").glob("trace_*.jsonl"))
    before = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    assert len(before) == 3  # question 1 in full, question 2 baseline only

    resumed = run_evaluation(
        _settings(tmp_path, corpus_dir, question_file, triple_cache, systems),
        FakeModelClient(),
        resume_trace=trace,
    )
    fresh = run_evaluation(
        _settings(tmp_path / "fresh", corpus_dir, question_file, triple_cache, systems),
        FakeModelClient(),
    )
    after = [
        json.loads(line)
        for line in resumed.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert resumed.trace_path == trace
    assert len(after) == 3 * len(systems)
    assert len({(r["question_id"], r["system"]) for r in after}) == len(after)
    assert {r["run_id"] for r in after} == {before[0]["run_id"]}

    def summary(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return {
                row["system"]: (row["acc"], row["recall"], row["total_cost_usd"])
                for row in csv.DictReader(stream)
            }

    assert summary(resumed.summary_path) == summary(fresh.summary_path)
    with resumed.detail_path.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_resume_refuses_a_trace_from_another_configuration(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    first = run_evaluation(
        _settings(tmp_path, corpus_dir, question_file, triple_cache, ("baseline",)),
        fake_client,
    )
    other = _settings(
        tmp_path, corpus_dir, question_file, triple_cache, ("baseline", "+KG")
    )
    with pytest.raises(ConfigurationError, match="different configuration"):
        run_evaluation(other, fake_client, resume_trace=first.trace_path)


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


def test_faithfulness_can_be_left_unjudged_and_is_never_a_zero(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    settings = _settings(
        tmp_path,
        corpus_dir,
        question_file,
        triple_cache,
        ("baseline",),
        judge_faithfulness=False,
    )
    outputs = run_evaluation(settings, fake_client)
    assert not [user for _, user, _ in fake_client.calls if "FAITHFUL" in user]
    judged = run_evaluation(
        replace(settings, judge_faithfulness=True, results_dir=tmp_path / "judged"),
        fake_client,
    )
    assert outputs.config_hash != judged.config_hash
    records = [
        json.loads(line)
        for line in outputs.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["metrics"]["faith"] is None for record in records)
    assert all(set(record["raw_judges"]) == {"accuracy"} for record in records)
    with outputs.summary_path.open(newline="", encoding="utf-8") as stream:
        summary = next(csv.DictReader(stream))
    assert summary["faith"] == "nan" and summary["acc"] == "1.0"
    with outputs.detail_path.open(newline="", encoding="utf-8") as stream:
        assert next(csv.DictReader(stream))["baseline_faith"] == ""


def test_unjudged_faithfulness_survives_a_resume(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    settings = _settings(
        tmp_path,
        corpus_dir,
        question_file,
        triple_cache,
        ("baseline",),
        judge_faithfulness=False,
    )
    outputs = run_evaluation(settings, fake_client)
    resumed = run_evaluation(settings, fake_client, resume_trace=outputs.trace_path)
    with resumed.summary_path.open(newline="", encoding="utf-8") as stream:
        assert next(csv.DictReader(stream))["faith"] == "nan"


def test_config_hash_ignores_faithfulness_at_its_default():
    base = {
        "name": "x",
        "corpus": "c",
        "corpus_dir": "d",
        "question_set": "q",
        "systems": ("baseline",),
        "generators": ("gpt4o-mini",),
        "pricing_snapshot": "p",
    }
    plain = ExperimentConfig(**base)
    identity = asdict(plain)
    del identity["judge_faithfulness"]
    assert plain.config_hash == content_hash(identity)
    assert (
        plain.config_hash
        != ExperimentConfig(**base, judge_faithfulness=False).config_hash
    )


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
        pricing_snapshot="configs/pricing_2026-10-20.json",
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
        pricing_snapshot="configs/pricing_2026-10-20.json",
        comparator_retrievals=".cache/hipporag/hotpotqa/retrievals.json",
    )
    settings = EvaluationSettings.from_config(
        config, "gpt4o-mini", pricing_captured_on="x", root=tmp_path
    )
    assert settings.comparator_file == tmp_path / config.comparator_retrievals


def test_passage_arm_report_scores_what_each_arm_would_retrieve(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    index = TextIndex.build(corpus_dir, fake_client, cache_dir=tmp_path / ".cache")
    ctx = RunContext(
        client=fake_client,
        model="",
        text_index=index,
        candidate_budget=5,
        graph=load_graph(corpus_dir, triple_cache),
        lexicon=chunk_lexicon(index),
    )
    arms = [arm for arm in PASSAGE_RETRIEVERS if arm != "+HippoRAG"]
    report = passage_arm_report(load_questions(question_file), ctx, arms)
    summary = report["summary"]
    assert list(summary) == arms
    assert summary["dense-top3"]["mean_passages"] == 3
    assert summary["baseline"]["mean_passages"] == 5
    assert summary["+random"]["mean_passages"] == 5
    for arm in arms:
        assert summary[arm]["complete"] <= summary[arm]["any_gold"] <= 3
    assert not fake_client.calls, "a retrieval report must not generate"
    first = report["outcomes"][0]
    assert first["baseline"]["sources"][:3] == first["dense-top3"]["sources"]


EQUAL_BUDGET_ARMS = (
    "dense-top3",
    "baseline",
    "bm25",
    "hybrid-rrf",
    "+KG",
    "+KGret",
    "+lexical",
    "+random",
)


def test_equal_budget_arms_run_end_to_end(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    settings = _settings(
        tmp_path, corpus_dir, question_file, triple_cache, EQUAL_BUDGET_ARMS
    )
    outputs = run_evaluation(settings, fake_client)
    with outputs.summary_path.open(newline="", encoding="utf-8") as stream:
        assert [row["system"] for row in csv.DictReader(stream)] == list(
            EQUAL_BUDGET_ARMS
        )
    traces = [
        json.loads(line)
        for line in outputs.trace_path.read_text(encoding="utf-8").splitlines()
    ]

    def text_items(system):
        return [
            [item for item in t["evidence"]["items"] if item["modality"] == "text"]
            for t in traces
            if t["system"] == system
        ]

    for system, count in (("dense-top3", 3), ("+KG", 3), ("baseline", 5)):
        assert {len(items) for items in text_items(system)} == {count}, system
    assert {len(items) for items in text_items("+random")} == {5}
    for system in ("bm25", "hybrid-rrf"):
        assert {
            item["selection_reason"] for items in text_items(system) for item in items
        } == {system}
    for system in ("+KGret", "+lexical", "+random"):
        for items in text_items(system):
            assert len(items) <= 5
            assert all(item["selection_reason"] == "dense" for item in items[:3])
    padded = [items[3:] for items in text_items("+random")]
    assert {item["selection_reason"] for extra in padded for item in extra} == {
        "random padding"
    }


def test_passage_arms_show_the_generator_no_marker_of_the_arm(
    tmp_path, corpus_dir, question_file, triple_cache, fake_client
):
    settings = _settings(
        tmp_path, corpus_dir, question_file, triple_cache, ("+KGret", "+lexical")
    )
    run_evaluation(settings, fake_client)
    prompts = [user for name, user, _ in fake_client.calls if name == "gpt4o-mini"]
    assert prompts
    assert not any("graph-bridged" in user or " | " in user for user in prompts)


def test_lexical_arms_need_their_index_and_graph_arms_their_graph(
    corpus_dir, fake_client, tmp_path
):
    index = TextIndex.build(corpus_dir, fake_client, cache_dir=tmp_path / ".cache")
    question = load_questions(_question_path(tmp_path))[0]
    ctx = RunContext(
        client=fake_client, model="gpt4o-mini", text_index=index, candidate_budget=5
    )
    with pytest.raises(ConfigurationError, match="BM25"):
        run_bm25(question, ctx, False)
    with pytest.raises(MissingEvidenceError, match="knowledge graph"):
        run_lexical(question, ctx, False)


def _question_path(tmp_path):
    path = tmp_path / "one_question.json"
    path.write_text(
        json.dumps(
            [
                {
                    "q": "Which receptor does compound Zeta bind?",
                    "source": "beta.txt",
                    "answer": "the kappa receptor",
                    "type": "text",
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


class _StubImageIndex:
    def __init__(self, captions):
        self.captions = captions


def _figure_question(tmp_path):
    path = tmp_path / "figure_questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "q": "What dose of compound Zeta does the figure show?",
                    "source": "fig_zeta.png",
                    "answer": "40 mg",
                    "type": "figure",
                }
            ]
        ),
        encoding="utf-8",
    )
    return load_questions(path)[0]


@pytest.mark.parametrize("use_pixels", [True, False])
def test_oracle_image_supplies_the_gold_crop_with_the_baseline_text(
    tmp_path, corpus_dir, fake_client, use_pixels
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "fig_zeta.png").write_bytes(b"png")
    (image_dir / "fig_other.png").write_bytes(b"png")
    ctx = RunContext(
        client=fake_client,
        model="gpt4o-mini",
        text_index=TextIndex.build(
            corpus_dir, fake_client, cache_dir=tmp_path / ".cache"
        ),
        candidate_budget=5,
        image_index=_StubImageIndex(
            {"fig_zeta.png": "Dose schedule", "fig_other.png": "Distractor plot"}
        ),
        image_dir=image_dir,
    )
    output = run_oracle_image(_figure_question(tmp_path), ctx, use_pixels)
    assert output.ranked_image_sources == ["fig_zeta.png"]
    assert len(output.text_passages) == 5
    _, user, images = fake_client.calls[-1]
    if use_pixels:
        assert output.image_paths == (image_dir / "fig_zeta.png",)
        assert images == (str(image_dir / "fig_zeta.png"),)
    else:
        assert output.image_paths == ()
        assert "(fig_zeta.png) Dose schedule" in user
        assert "Distractor plot" not in user
    reasons = {
        item.selection_reason
        for item in output.evidence_bundle(5).items
        if item.modality == "image"
    }
    assert reasons == {"oracle-gold (sent)"}


def test_oracle_image_refuses_a_question_without_a_gold_crop(
    tmp_path, corpus_dir, fake_client
):
    ctx = RunContext(
        client=fake_client,
        model="gpt4o-mini",
        text_index=TextIndex.build(
            corpus_dir, fake_client, cache_dir=tmp_path / ".cache"
        ),
        candidate_budget=5,
        image_index=_StubImageIndex({}),
        image_dir=tmp_path,
    )
    with pytest.raises(MissingEvidenceError, match="no gold image"):
        run_oracle_image(load_questions(_question_path(tmp_path))[0], ctx, True)


class _CappedClient(FakeModelClient):
    """Leaves the generator's answer to one question empty at the legacy cap."""

    def __init__(self, question, **kwargs):
        super().__init__(**kwargs)
        self.question = question
        self.generator_caps = []

    def call(
        self, name, *, user, system="", images=(), max_tokens=800, temperature=0.0
    ):
        result = super().call(
            name,
            user=user,
            system=system,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if name != "gpt4o-mini":
            return result
        self.generator_caps.append(max_tokens)
        if max_tokens <= LEGACY_GENERATOR_MAX_TOKENS and self.question in user:
            return replace(result, text="", output_tokens=max_tokens)
        return result

    def generator_prompts(self):
        return [user for name, user, _ in self.calls if name == "gpt4o-mini"]


REGENERATED_SYSTEMS = (
    "baseline",
    "+KG",
    "+KGret",
    "control:closed-book",
    "control:oracle",
)


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _legacy_run(tmp_path, corpus_dir, question_file, triple_cache, monkeypatch):
    """A finished run at the legacy cap whose second question was cut off."""
    target = load_questions(question_file)[1]
    monkeypatch.setattr(evaluate, "GENERATOR_MAX_TOKENS", LEGACY_GENERATOR_MAX_TOKENS)
    client = _CappedClient(target.question)
    settings = _settings(
        tmp_path, corpus_dir, question_file, triple_cache, REGENERATED_SYSTEMS
    )
    outputs = run_evaluation(settings, client)
    monkeypatch.setattr(evaluate, "GENERATOR_MAX_TOKENS", GENERATOR_MAX_TOKENS)
    return target, client, settings, outputs


def test_regeneration_answers_capped_records_again_from_their_evidence(
    tmp_path, corpus_dir, question_file, triple_cache, monkeypatch
):
    target, first, settings, legacy = _legacy_run(
        tmp_path, corpus_dir, question_file, triple_cache, monkeypatch
    )
    before = _records(legacy.trace_path)
    assert {r["question_id"] for r in before if r["answer"] == ""} == {target.id}
    other = next(r for r in before if r["question_id"] != target.id)
    judgements = judgements_path(legacy.trace_path)
    entries = [
        (target.id, "baseline", "stability"),
        (target.id, "baseline", "sufficiency"),
        (other["question_id"], "baseline", "stability"),
    ]
    judgements.write_text(
        "".join(
            json.dumps(
                {"key": str(i), "question_id": q, "system": s, "pass": p, "value": 1}
            )
            + "\n"
            for i, (q, s, p) in enumerate(entries)
        ),
        encoding="utf-8",
    )

    second = _CappedClient(target.question)
    amended = run_evaluation(
        settings, second, resume_trace=legacy.trace_path, regenerate_capped=True
    )
    after = _records(amended.trace_path)
    assert amended.trace_path == legacy.trace_path
    assert {r["config_hash"] for r in after} == {amended.config_hash}
    assert amended.config_hash != legacy.config_hash
    for old, new in zip(before, after, strict=True):
        assert (new["question_id"], new["system"]) == (
            old["question_id"],
            old["system"],
        )
        assert new["evidence"] == old["evidence"]
        if old["question_id"] == target.id:
            assert new["answer"] == second.answer
            assert new["regeneration"]["previous"]["answer"] == ""
            assert new["regeneration"]["previous"]["output_tokens"] == 800
            assert new["regeneration"]["max_tokens"] == GENERATOR_MAX_TOKENS
        else:
            assert {**new, "config_hash": old["config_hash"]} == old
    # Only the capped answers were generated again, from the very same prompts.
    target_prompts = [p for p in first.generator_prompts() if target.question in p]
    assert second.generator_prompts() == target_prompts
    assert set(second.generator_caps) == {GENERATOR_MAX_TOKENS}

    with amended.summary_path.open(newline="", encoding="utf-8") as stream:
        out_tokens = {
            row["system"]: row["mean_out_tok"] for row in csv.DictReader(stream)
        }
    assert set(out_tokens.values()) == {"8.0"}
    kept = [
        json.loads(line)["key"]
        for line in judgements.read_text(encoding="utf-8").splitlines()
    ]
    assert kept == ["1", "2"]  # the replaced answer's stability verdict is gone
    backup = legacy.trace_path.parent / "pre_regeneration"
    assert _records(backup / legacy.trace_path.name) == before
    assert (backup / judgements.name).is_file()
    assert not list(legacy.trace_path.parent.glob("regenerated_*"))

    third = _CappedClient(target.question)
    again = run_evaluation(
        settings, third, resume_trace=legacy.trace_path, regenerate_capped=True
    )
    assert third.generator_prompts() == []
    assert _records(again.trace_path) == after


def test_an_interrupted_regeneration_resumes_where_it_stopped(
    tmp_path, corpus_dir, question_file, triple_cache, monkeypatch
):
    target, _, settings, legacy = _legacy_run(
        tmp_path, corpus_dir, question_file, triple_cache, monkeypatch
    )
    # One generator and two judge calls per answer: the eighth call fails.
    with pytest.raises(ProviderError):
        run_evaluation(
            settings,
            _CappedClient(target.question, fail_after_calls=7),
            resume_trace=legacy.trace_path,
            regenerate_capped=True,
        )
    progress = next(legacy.trace_path.parent.glob("regenerated_*.jsonl"))
    assert len(_records(progress)) == 2
    assert {r["config_hash"] for r in _records(legacy.trace_path)} == {
        legacy.config_hash
    }

    finish = _CappedClient(target.question)
    run_evaluation(
        settings, finish, resume_trace=legacy.trace_path, regenerate_capped=True
    )
    assert len(finish.generator_prompts()) == len(REGENERATED_SYSTEMS) - 2
    regenerated = [r for r in _records(legacy.trace_path) if r.get("regeneration")]
    assert len(regenerated) == len(REGENERATED_SYSTEMS)


def test_a_legacy_cap_trace_is_neither_resumed_nor_regenerated_elsewhere(
    tmp_path, corpus_dir, question_file, triple_cache, monkeypatch
):
    _, _, settings, legacy = _legacy_run(
        tmp_path, corpus_dir, question_file, triple_cache, monkeypatch
    )
    with pytest.raises(ConfigurationError, match="different configuration"):
        run_evaluation(settings, FakeModelClient(), resume_trace=legacy.trace_path)
    other = _settings(tmp_path, corpus_dir, question_file, triple_cache, ("baseline",))
    with pytest.raises(ConfigurationError, match="cannot be regenerated"):
        run_evaluation(
            other,
            FakeModelClient(),
            resume_trace=legacy.trace_path,
            regenerate_capped=True,
        )


def test_image_arms_are_not_replayed(question_file):
    question = load_questions(question_file)[0]
    with pytest.raises(ConfigurationError, match="rerun it"):
        replay("+multimodal", question, None, {"items": []})
