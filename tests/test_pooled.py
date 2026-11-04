import json

import numpy as np
import pytest

from multimodal_graph_rag.cli import main
from multimodal_graph_rag.errors import ConfigurationError
from multimodal_graph_rag.evaluation.pooled import (
    adjust_family,
    compare,
    load_outcomes,
)
from multimodal_graph_rag.evaluation.statistics import (
    access_use_split,
    cluster_bootstrap_delta,
    sign_flip_test,
)
from multimodal_graph_rag.schemas import load_questions

QUESTIONS = [
    {
        "q": f"Question {i}?",
        "source": f"doc{i}.txt",
        "answer": f"answer {i}",
        "type": "text",
    }
    for i in range(6)
]


def _record(qid, system, generator, correct, complete, recall=None):
    return {
        "question_id": qid,
        "system": system,
        "answer": "answer" if correct else "I don't know",
        "metrics": {
            "acc": int(correct),
            "complete": int(complete),
            "recall": float(complete if recall is None else recall),
            "faith": 1,
        },
        "model_versions": {generator: f"vendor/{generator}"},
    }


@pytest.fixture
def traces(tmp_path):
    """Two generators; the arm completes and answers questions 0-2 that the
    baseline misses, and loses question 5."""
    question_file = tmp_path / "questions.json"
    question_file.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    questions = load_questions(question_file)
    paths = []
    for generator in ("gen-a", "gen-b"):
        rows = []
        for i, question in enumerate(questions):
            base_correct = i >= 3
            arm_correct = i != 5
            rows.append(
                _record(question.id, "baseline", generator, base_correct, base_correct)
            )
            rows.append(
                _record(question.id, "arm", generator, arm_correct, arm_correct)
            )
            rows.append(
                _record(question.id, "control:closed-book", generator, False, False)
            )
        path = tmp_path / f"trace_{generator}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        paths.append(path)
    return question_file, questions, paths


def test_sign_flip_test_is_one_without_differences_and_small_for_a_shift():
    assert sign_flip_test([0.0] * 20, iterations=500) == 1.0
    assert sign_flip_test([1.0] * 20, iterations=2000, seed=1) < 0.01


def test_cluster_bootstrap_keeps_a_question_together():
    reference = [[0, 0], [1, 1], [0, 1]]
    treatment = [[1, 1], [1, 1], [0, 1]]
    estimate = cluster_bootstrap_delta(reference, treatment, iterations=500, seed=2)
    assert estimate.delta == pytest.approx(1 / 3)
    assert estimate.samples == 3
    assert 0.0 <= estimate.lower <= estimate.delta <= estimate.upper <= 1.0


def test_access_use_split_sums_to_the_difference_both_ways():
    reference = (
        [0, 1, 0, 1, 1, 0],
        ["none", "complete", "partial", "complete", "partial", "none"],
    )
    treatment = (
        [1, 1, 0, 1, 1, 1],
        ["complete", "complete", "partial", "complete", "complete", "none"],
    )
    split = access_use_split(*reference, *treatment)
    assert split.access + split.use == pytest.approx(split.delta)
    assert split.symmetric_access + split.symmetric_use == pytest.approx(split.delta)
    reverse = access_use_split(*treatment, *reference)
    assert reverse.symmetric_access == pytest.approx(-split.symmetric_access)


def test_compare_pools_generators_with_the_question_as_the_unit(traces):
    _, questions, paths = traces
    outcomes = load_outcomes(paths, questions)
    assert outcomes.generators == ("gen-a", "gen-b")
    report = compare(
        outcomes, "baseline", splits=[("arm", "baseline")], iterations=500, seed=3
    )
    arm = report["systems"]["arm"]["differences"]["acc"]
    assert arm["delta"] == pytest.approx(2 / 6)
    assert (arm["wins"], arm["losses"]) == (6, 2)
    assert 0 < arm["p"] <= 1 and arm["p_holm"] >= arm["p"]
    # controls are reported but not part of the Holm family
    assert (
        "p_holm" not in report["systems"]["control:closed-book"]["differences"]["acc"]
    )
    split = report["splits"]["arm vs baseline"]["acc"]
    assert split["access"][0] + split["use"][0] == pytest.approx(split["delta"][0])
    assert report["per_generator"]["arm"]["gen-b"]["acc"] == pytest.approx(5 / 6)


def test_a_family_spanning_two_reports_is_adjusted_together(traces):
    _, questions, paths = traces
    outcomes = load_outcomes(paths, questions)
    first = compare(outcomes, "baseline", iterations=200, seed=4)
    second = compare(outcomes, "baseline", iterations=200, seed=4)
    alone = first["systems"]["arm"]["differences"]["acc"]["p_holm"]
    adjust_family([first, second])
    together = first["systems"]["arm"]["differences"]["acc"]["p_holm"]
    p = first["systems"]["arm"]["differences"]["acc"]["p"]
    assert alone == pytest.approx(p)  # one test per report alone
    assert together == pytest.approx(min(1.0, 2 * p))  # two tests in the family


def test_containment_is_rescored_from_the_answers(traces):
    _, questions, paths = traces
    outcomes = load_outcomes(paths, questions)
    # stored answers are "answer" or "I don't know"; references are "answer i"
    assert not np.any(outcomes.values["arm"]["contains"])


def test_missing_outcomes_are_refused(traces):
    _, questions, paths = traces
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    paths[0].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="missing"):
        load_outcomes(paths, questions)


def test_cli_excludes_questions_without_a_reference(traces, tmp_path, capsys):
    question_file, _, paths = traces
    data = json.loads(question_file.read_text(encoding="utf-8"))
    data.append(
        {"q": "Unscorable?", "source": "doc9.txt", "answer": "...", "type": "text"}
    )
    question_file.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "report.json"
    code = main(
        [
            "pooled-analysis",
            *map(str, paths),
            "--questions",
            str(question_file),
            "--split",
            "arm=baseline",
            "--iterations",
            "200",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert "excluded 1 question" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["n_questions"] == 6
