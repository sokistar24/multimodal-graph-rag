import json

import pytest

from multimodal_graph_rag.errors import ConfigurationError
from multimodal_graph_rag.evaluation.audits import (
    figure_integrity_rows,
    paired_accuracy,
    sample_triple_audit,
)


def test_triple_audit_sample_is_deterministic_and_unique(triple_cache):
    first = sample_triple_audit(triple_cache, "test", 6, seed=1)
    second = sample_triple_audit(triple_cache, "test", 6, seed=1)
    assert first == second
    assert len({(r["subject"], r["relation"], r["object"]) for r in first}) == 6
    with pytest.raises(ConfigurationError):
        sample_triple_audit(triple_cache, "test", 100, seed=1)


def test_figure_integrity_flags_caption_and_corpus_leaks(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "1234.5678v1.txt").write_text(
        "The accuracy reached 91.5 percent on the test set.", encoding="utf-8"
    )
    captions = tmp_path / "captions.json"
    captions.write_text(
        json.dumps({"1234.5678v1-Figure2-1.png": "Bar chart of Macrophage counts."})
    )
    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps(
            [
                {
                    "q": "What accuracy?",
                    "source": "1234.5678v1-Figure2-1.png",
                    "answer": "91.5",
                    "type": "figure",
                },
                {
                    "q": "Which cell?",
                    "source": "1234.5678v1-Figure2-1.png",
                    "answer": "Macrophage",
                    "type": "figure",
                },
            ]
        )
    )
    rows = figure_integrity_rows(questions, captions, corpus)
    assert (
        rows[0]["corpus_exact"] is True
        and rows[0]["audit_status"] == "flagged_for_human_review"
    )
    assert rows[1]["caption_exact"] is True


def test_paired_accuracy_requires_both_columns():
    rows = [
        {"baseline_acc": "1", "+KGret_acc": "0"},
        {"baseline_acc": "0", "+KGret_acc": "1"},
    ]
    assert paired_accuracy(rows, "baseline", "+KGret") == ([1, 0], [0, 1])
    with pytest.raises(ConfigurationError):
        paired_accuracy(rows, "baseline", "+KG")
