import json

import pytest

from multimodal_graph_rag.errors import ConfigurationError
from multimodal_graph_rag.questions.spiqa_native import convert_spiqa_native
from multimodal_graph_rag.schemas import load_questions

GOLD = [
    {
        "question": "Which dataset has the most 4-hop triples?",
        "answer": "Bing-News.",
        "explanation": "The table reports 6,322,548 for Bing-News.",
        "reference": "1803.03467v4-Table1-1.png",
        "paper_id": "1803.03467v4",
    },
    {
        "question": "What does the figure show about common raters?",
        "answer": "Items with common raters have more common neighbours.",
        "explanation": "The bars are consistently higher.",
        "reference": "1803.03467v4-Figure4-1.png",
        "paper_id": "1803.03467v4",
    },
    {
        # The crop for this one was never ingested, so it must be skipped.
        "question": "What is the accuracy on the held-out split?",
        "answer": "91.2 percent.",
        "reference": "9999.99999v1-Figure1-1.png",
        "paper_id": "9999.99999v1",
    },
    {
        "question": "",
        "answer": "",
        "reference": "",
        "paper_id": "",
    },
]
CAPTIONS = {
    # Leaks the answer verbatim.
    "1803.03467v4-Table1-1.png": "A table of hop counts in which Bing-News. leads.",
    # Does not leak it.
    "1803.03467v4-Figure4-1.png": "A bar chart of neighbour counts by hop distance.",
}


@pytest.fixture
def gold_files(tmp_path):
    gold = tmp_path / "spiqa_gold_qa.json"
    gold.write_text(json.dumps(GOLD), encoding="utf-8")
    captions = tmp_path / "captions.json"
    captions.write_text(json.dumps(CAPTIONS), encoding="utf-8")
    return gold, captions


def test_conversion_keeps_human_authorship_and_screens_leakage(tmp_path, gold_files):
    gold, captions = gold_files
    out = tmp_path / "questions_spiqa_native.json"
    report = convert_spiqa_native(gold, captions, out)

    assert report.written == 2
    assert report.skipped_missing_crop == 1
    assert report.skipped_incomplete == 1
    # One question says "figure"; the native set is not filtered for that.
    assert report.names_its_modality == 1

    records = load_questions(out)
    assert {r.construction_method for r in records} == {"spiqa_native_human"}
    assert all(not r.graph_seeded for r in records)
    # The reference is a crop, so retrieval is scored on the image ranking.
    assert all(r.is_visual for r in records)

    by_source = {r.gold_sources[0]: r for r in records}
    assert (
        by_source["1803.03467v4-Table1-1.png"].leakage_status
        == "flagged_for_human_review"
    )
    assert (
        by_source["1803.03467v4-Figure4-1.png"].leakage_status == "screened_unflagged"
    )


def test_ids_are_stable_across_conversions(tmp_path, gold_files):
    gold, captions = gold_files
    first = convert_spiqa_native(gold, captions, tmp_path / "a.json")
    second = convert_spiqa_native(gold, captions, tmp_path / "b.json")
    assert first.written == second.written
    assert [r.id for r in load_questions(tmp_path / "a.json")] == [
        r.id for r in load_questions(tmp_path / "b.json")
    ]


def test_exclude_flagged_drops_the_caption_leak(tmp_path, gold_files):
    gold, captions = gold_files
    out = tmp_path / "clean.json"
    convert_spiqa_native(gold, captions, out, exclude_flagged=True)
    records = load_questions(out)
    assert [r.gold_sources[0] for r in records] == ["1803.03467v4-Figure4-1.png"]


def test_target_larger_than_the_pool_is_refused(tmp_path, gold_files):
    gold, captions = gold_files
    with pytest.raises(ConfigurationError, match="only 2 are available"):
        convert_spiqa_native(gold, captions, tmp_path / "x.json", target=5)


def test_no_matching_crop_is_an_error_not_an_empty_set(tmp_path, gold_files):
    gold, _ = gold_files
    empty = tmp_path / "other_captions.json"
    empty.write_text(json.dumps({"unrelated.png": "nothing"}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="no SPIQA gold QA item"):
        convert_spiqa_native(gold, empty, tmp_path / "y.json")
