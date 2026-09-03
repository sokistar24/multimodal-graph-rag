import json

import pytest

from multimodal_graph_rag.errors import ConfigurationError
from multimodal_graph_rag.schemas import (
    EvidenceBundle,
    EvidenceItem,
    QuestionRecord,
    load_questions,
    write_questions,
)


def test_legacy_cross_question_marks_graph_seeded_and_keeps_both_sources():
    question = QuestionRecord.from_legacy(
        {
            "q": "How do two papers use the same entity?",
            "answer": "In different ways.",
            "source": "a.txt",
            "source2": "b.txt",
            "type": "multihop_cross",
            "entity": "shared entity",
        }
    )
    assert question.gold_sources == ("a.txt", "b.txt")
    assert question.graph_seeded is True
    assert not question.is_visual
    assert question.text_gold_sources == ("a.txt", "b.txt")


def test_visual_questions_are_scored_on_images():
    question = QuestionRecord.from_legacy(
        {"q": "q?", "answer": "a", "source": "crop.png", "type": "figure"}
    )
    assert question.is_visual
    assert question.evidence_requirement == "visual"


def test_evidence_bundle_requires_every_gold_source():
    bundle = EvidenceBundle(
        items=(EvidenceItem("text", "a.txt", 1), EvidenceItem("text", "b.txt", 2)),
        candidate_budget=2,
    )
    assert bundle.is_complete(["a.txt", "b.txt"])
    assert not bundle.is_complete(["a.txt", "c.txt"])


def test_load_questions_supports_both_layouts_and_rejects_duplicates(
    tmp_path, question_file
):
    legacy = load_questions(question_file)
    assert len(legacy) == 3
    versioned = tmp_path / "versioned.json"
    digest = write_questions(legacy, versioned, source_file="legacy")
    assert json.loads(versioned.read_text())["questions_sha256"] == digest
    assert [q.id for q in load_questions(versioned)] == [q.id for q in legacy]
    duplicate = tmp_path / "dup.json"
    duplicate.write_text(
        json.dumps([{"q": "a?", "answer": "b", "source": "s.txt"}] * 2)
    )
    with pytest.raises(ConfigurationError, match="duplicate"):
        load_questions(duplicate)
