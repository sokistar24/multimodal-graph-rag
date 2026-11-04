import json

import pytest
from conftest import FakeModelClient

from multimodal_graph_rag.errors import ConfigurationError
from multimodal_graph_rag.evaluation.calibration import (
    calibration_items,
    run_calibration,
)
from multimodal_graph_rag.schemas import load_questions


def _questions(tmp_path, answers):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "q": f"Question {i}?",
                    "source": f"d{i}.txt",
                    "answer": a,
                    "type": "text",
                }
                for i, a in enumerate(answers)
            ]
        ),
        encoding="utf-8",
    )
    return load_questions(path)


def test_each_question_gets_its_reference_and_a_different_decoy(tmp_path):
    questions = _questions(tmp_path, ["yes", "yes", "no", "Paris", "..."])
    items = calibration_items(questions)
    # the unscorable "..." question is left out
    assert len(items) == 8
    decoys = {i.question.answer: i.answer for i in items if i.kind == "decoy"}
    assert decoys == {"yes": "no", "no": "Paris", "Paris": "yes"}
    assert all(i.expected == (1 if i.kind == "reference" else 0) for i in items)


def test_a_set_with_one_reference_cannot_be_calibrated(tmp_path):
    with pytest.raises(ConfigurationError):
        calibration_items(_questions(tmp_path, ["yes", "Yes."]))


def test_calibration_resumes_and_reports_both_error_rates(tmp_path):
    items = calibration_items(_questions(tmp_path, ["yes", "no", "Paris"]))
    output = tmp_path / "calibration.jsonl"
    first = FakeModelClient()  # its judge accepts everything
    summary = run_calibration(items, first, "deepseek", output)
    assert summary["items"] == 6
    assert summary["false_negative_rate"] == 0.0
    assert summary["false_positive_rate"] == 1.0
    again = FakeModelClient()
    assert run_calibration(items, again, "deepseek", output) == summary
    assert again.calls == []
