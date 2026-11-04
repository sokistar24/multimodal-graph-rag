"""Judging passes and evidence states over saved run records."""

from multimodal_graph_rag.evaluation.posthoc import (
    JudgementStore,
    assign_states,
    gold_received,
    judge_context,
    plan_tasks,
    run_task,
    stratified_sample,
)
from multimodal_graph_rag.schemas import QuestionRecord

QUESTIONS = {
    "q1": QuestionRecord(
        id="q1",
        question="Who wrote it?",
        answer="Margery Williams",
        gold_sources=("a.txt", "b.txt"),
        question_type="hotpot_bridge",
    ),
    "q2": QuestionRecord(
        id="q2",
        question="Which node is latent?",
        answer="eta",
        gold_sources=("fig.png",),
        question_type="figure",
    ),
}


def _text(source, content="text"):
    return {
        "modality": "text",
        "source": source,
        "content": content,
        "selection_reason": "dense",
    }


def _image(source, sent, caption="a caption"):
    return {
        "modality": "image",
        "source": source,
        "content": caption,
        "selection_reason": "clip (sent)" if sent else "clip (scored only)",
    }


def _record(qid, system, items, acc, answer="Margery Williams"):
    return {
        "run_id": "run1",
        "question_id": qid,
        "system": system,
        "answer": answer,
        "evidence": {"items": items},
        "raw_judges": {"accuracy": {"value": acc}},
    }


def test_context_and_gold_follow_what_reached_the_generator():
    record = _record(
        "q2",
        "+multimodal",
        [
            _text("p.txt", "passage"),
            _image("fig.png", sent=False),
            _image("other.png", sent=True, caption="other caption"),
        ],
        acc=0,
    )
    assert judge_context(record) == "passage\nother caption"
    # the gold crop was scored but not sent, so it did not reach the generator
    assert gold_received(record, QUESTIONS["q2"]) == 0
    text = _record("q1", "baseline", [_text("a.txt"), _text("c.txt")], acc=1)
    assert gold_received(text, QUESTIONS["q1"]) == 1


def test_stratified_sample_is_deterministic_and_covers_every_system():
    records = [
        _record(f"q{i}", s, [], 1) for i in range(20) for s in ("baseline", "+KG")
    ]
    first = stratified_sample(records, 0.2, seed=7)
    assert first == stratified_sample(records, 0.2, seed=7)
    assert sorted(r["system"] for r in first).count("baseline") == 4
    assert sorted(r["system"] for r in first).count("+KG") == 4


def test_sufficiency_targets_correct_answers_without_all_gold(tmp_path):
    store = JudgementStore(tmp_path / "judgements.jsonl")
    records = [
        _record("q1", "baseline", [_text("a.txt")], acc=1),  # correct, incomplete
        _record("q1", "+KGret", [_text("a.txt"), _text("b.txt")], acc=1),  # complete
        _record("q1", "+random", [_text("a.txt")], acc=0),  # incorrect
        _record("q1", "control:closed-book", [], acc=1),  # no context
    ]
    tasks = plan_tasks("sufficiency", records, QUESTIONS, store)
    assert [t.record["system"] for t in tasks] == ["baseline"]


def test_adjudication_only_where_the_two_judges_disagree(tmp_path):
    store = JudgementStore(tmp_path / "judgements.jsonl")
    records = [
        _record("q1", "baseline", [], acc=1),
        _record("q1", "+KG", [], acc=0),
    ]
    for record, value in ((records[0], 1), (records[1], 1)):
        store.add(
            {
                "key": f"run1|q1|{record['system']}|second-accuracy|0",
                "question_id": "q1",
                "system": record["system"],
                "pass": "second-accuracy",
                "value": value,
            }
        )
    tasks = plan_tasks("adjudicate", records, QUESTIONS, store)
    assert [t.record["system"] for t in tasks] == ["+KG"]


def test_finished_tasks_are_skipped_on_resume(fake_client, tmp_path):
    store = JudgementStore(tmp_path / "judgements.jsonl")
    records = [_record("q1", "baseline", [_text("a.txt")], acc=1)]
    task = plan_tasks("second-accuracy", records, QUESTIONS, store)[0]
    store.add(
        run_task(
            task, QUESTIONS["q1"], fake_client, "deepseek-v4-pro", None, "captions"
        )
    )
    reopened = JudgementStore(tmp_path / "judgements.jsonl")
    assert plan_tasks("second-accuracy", records, QUESTIONS, reopened) == []


def test_images_go_to_the_judge_only_in_pixel_mode(fake_client, tmp_path):
    store = JudgementStore(tmp_path / "judgements.jsonl")
    record = _record(
        "q2", "+multimodal", [_image("fig2.png", sent=True)], acc=1, answer="eta"
    )
    task = plan_tasks("sufficiency", [record], QUESTIONS, store)[0]
    run_task(task, QUESTIONS["q2"], fake_client, "gpt-6.1-sol", tmp_path, "pixels")
    assert fake_client.calls[-1][2] == (str(tmp_path / "fig2.png"),)
    run_task(task, QUESTIONS["q2"], fake_client, "gpt-6.1-sol", tmp_path, "captions")
    assert fake_client.calls[-1][2] == ()


def test_states_wait_for_sufficiency_and_use_the_closed_book_arm(tmp_path):
    store = JudgementStore(tmp_path / "judgements.jsonl")
    records = [
        _record("q1", "baseline", [_text("a.txt")], acc=1),
        _record("q1", "+KGret", [_text("a.txt"), _text("b.txt")], acc=1),
        _record("q1", "+random", [_text("c.txt")], acc=0),
        _record("q1", "control:closed-book", [], acc=0),
    ]
    available = {"a.txt", "b.txt", "c.txt"}
    states = {
        r["system"]: r["state"]
        for r in assign_states(records, QUESTIONS, store, available_sources=available)
    }
    assert states == {
        "baseline": "pending_sufficiency",
        "+KGret": "complete_retrieval",
        "+random": "retrieval_failure",
    }
    store.add(
        {
            "key": "run1|q1|baseline|sufficiency|0",
            "question_id": "q1",
            "system": "baseline",
            "pass": "sufficiency",
            "value": 1,
        }
    )
    states = {
        r["system"]: r["state"]
        for r in assign_states(records, QUESTIONS, store, available_sources=available)
    }
    assert states["baseline"] == "non_gold_supported_answer"
