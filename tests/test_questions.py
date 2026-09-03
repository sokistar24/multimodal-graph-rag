import json

from multimodal_graph_rag.questions import clean_question_file
from multimodal_graph_rag.questions.validators import (
    has_orphan_reference,
    leaks_modality,
    parse_json_array,
    rejection_reason,
)
from multimodal_graph_rag.schemas import load_questions


def test_validators_catch_each_failure_mode():
    assert leaks_modality("What is shown in the figure?")
    assert has_orphan_reference("How many people attended the workshop?")
    assert not has_orphan_reference("How many attended the 2011 WHO Geneva workshop?")
    assert (
        rejection_reason("What does this work propose?")
        == "container reference or source identifier"
    )
    assert (
        rejection_reason("What is the arXiv paper 1809.00263?")
        == "container reference or source identifier"
    )
    assert rejection_reason("Which vaccine is used in China for leptospirosis?") is None
    assert rejection_reason("x" * 221) == "over-length"


def test_parse_json_array_tolerates_fences():
    assert parse_json_array('```json\n[{"q": "a", "answer": "b"}]\n```') == [
        {"q": "a", "answer": "b"}
    ]
    assert parse_json_array("nonsense") is None
    assert parse_json_array('{"q": 1}') is None


def test_clean_question_file_rewrites_versioned_layout(tmp_path):
    path = tmp_path / "questions_x.json"
    path.write_text(
        json.dumps(
            [
                {
                    "q": "What is depicted in the chart?",
                    "source": "a.png",
                    "answer": "x",
                    "type": "figure",
                },
                {
                    "q": "Which vaccine is used in China for leptospirosis?",
                    "source": "b.txt",
                    "answer": "y",
                    "type": "text",
                },
            ]
        ),
        encoding="utf-8",
    )
    report = clean_question_file(path)
    assert report.kept == 1
    assert report.removed[0][0] == "modality leak"
    reloaded = load_questions(path)
    assert [q.question_type for q in reloaded] == ["text"]
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
