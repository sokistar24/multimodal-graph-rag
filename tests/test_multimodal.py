import csv
import json

import numpy as np
import pytest

from multimodal_graph_rag.cli import main
from multimodal_graph_rag.errors import ConfigurationError
from multimodal_graph_rag.evaluation.multimodal import (
    CAPTIONS,
    CLOSED_BOOK,
    ORACLE_CAPTION,
    ORACLE_IMAGE,
    PIXELS,
    apply_adjudication,
    closed_book_recoverable,
    filtering_change,
    gain_difference,
    gains,
    gold_sent,
    merge_caption_arms,
    restrict,
    text_recoverable,
)
from multimodal_graph_rag.evaluation.pooled import load_outcomes
from multimodal_graph_rag.schemas import load_questions

GENERATORS = ("gen-a", "gen-b")


def _question_file(path, prefix, count):
    data = [
        {
            "q": f"{prefix} question {i}?",
            "source": f"{prefix}{i}.png",
            "answer": f"answer {i}",
            "type": "image",
        }
        for i in range(count)
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _record(question, system, generator, correct, *, sent=None, scored=()):
    items = [
        {
            "modality": "image",
            "source": name,
            "selection_reason": "clip (sent)" if name == sent else "clip (scored only)",
            "content": "",
        }
        for name in scored
    ]
    return {
        "question_id": question.id,
        "system": system,
        "answer": "answer" if correct else "unknown",
        "metrics": {
            "acc": int(correct),
            "complete": int(question.gold_sources[0] in scored),
            "recall": float(question.gold_sources[0] in scored),
            "faith": None,
        },
        "model_versions": {generator: f"vendor/{generator}"},
        "evidence": {"items": items},
    }


def _write_trace(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _pixel_traces(tmp_path, stem, questions, pixel_correct):
    """Text-only never right, closed-book right on question 0, oracle always right.

    The top crop is gold on question 0 only, and gold is ranked second on
    question 1, so it is scored there without being sent.
    """
    paths = []
    for generator in GENERATORS:
        rows = []
        for i, question in enumerate(questions):
            gold = question.gold_sources[0]
            if i == 0:
                sent, scored = gold, (gold, "other.png")
            elif i == 1:
                sent, scored = "other.png", ("other.png", gold)
            else:
                sent, scored = "other.png", ("other.png",)
            rows += [
                _record(question, "baseline", generator, False),
                _record(question, CLOSED_BOOK, generator, i == 0),
                _record(
                    question, ORACLE_IMAGE, generator, True, sent=gold, scored=(gold,)
                ),
                _record(
                    question,
                    PIXELS,
                    generator,
                    pixel_correct(i),
                    sent=sent,
                    scored=scored,
                ),
            ]
        paths.append(_write_trace(tmp_path / f"trace_{stem}_{generator}.jsonl", rows))
    return paths


def _caption_traces(tmp_path, questions):
    """Written in the other generator order, which the merge has to line up."""
    paths = []
    for generator in reversed(GENERATORS):
        rows = []
        for i, question in enumerate(questions):
            correct = generator == "gen-a" and i == 2
            rows += [
                _record(question, PIXELS, generator, correct),
                _record(question, ORACLE_IMAGE, generator, generator == "gen-a"),
            ]
        paths.append(_write_trace(tmp_path / f"trace_captions_{generator}.jsonl", rows))
    return paths


@pytest.fixture
def screened(tmp_path):
    path = _question_file(tmp_path / "screened.json", "s", 6)
    questions = load_questions(path)
    return path, questions, _pixel_traces(tmp_path, "s", questions, lambda i: i < 3)


def test_caption_arms_join_the_pixel_run_lined_up_by_generator(screened, tmp_path):
    _, questions, paths = screened
    pixels = load_outcomes(paths, questions)
    merged = merge_caption_arms(
        pixels, load_outcomes(_caption_traces(tmp_path, questions), questions)
    )
    assert set(merged.values) == set(pixels.values) | {CAPTIONS, ORACLE_CAPTION}
    column = merged.generators.index("gen-a")
    assert merged.values[CAPTIONS]["acc"][2, column] == 1
    assert merged.values[CAPTIONS]["acc"][2, 1 - column] == 0
    assert merged.values[ORACLE_CAPTION]["acc"][:, column].all()


def test_merging_refuses_a_caption_run_of_other_questions(screened, tmp_path):
    _, questions, paths = screened
    pixels = load_outcomes(paths, questions)
    fewer = questions[:3]
    captions = load_outcomes(_caption_traces(tmp_path, fewer), fewer)
    with pytest.raises(ConfigurationError, match="different questions"):
        merge_caption_arms(pixels, captions)


def test_adjudication_replaces_only_the_verdict_it_covers(screened):
    _, questions, paths = screened
    outcomes = load_outcomes(paths, questions)
    target = paths[outcomes.generators.index("gen-b")]
    judgements = target.with_name(target.name.replace("trace_", "judgements_", 1))
    entries = [
        # the primary judge said wrong here, so this one changes the outcome
        (questions[4].id, "baseline", 1),
        # the primary judge already said right here, so this one does not
        (questions[0].id, PIXELS, 1),
    ]
    judgements.write_text(
        "".join(
            json.dumps(
                {
                    "key": f"{qid}|{system}",
                    "question_id": qid,
                    "system": system,
                    "pass": "adjudicate",
                    "value": value,
                }
            )
            + "\n"
            for qid, system, value in entries
        ),
        encoding="utf-8",
    )
    before = outcomes.values["baseline"]["acc"].sum()
    assert apply_adjudication(outcomes, paths) == 1
    assert outcomes.values["baseline"]["acc"].sum() == before + 1
    column = outcomes.generators.index("gen-b")
    assert outcomes.values["baseline"]["acc"][4, column] == 1
    assert outcomes.values["baseline"]["acc"][4, 1 - column] == 0


def test_restrict_keeps_only_the_named_questions(screened):
    _, questions, paths = screened
    outcomes = load_outcomes(paths, questions)
    kept = restrict(outcomes, [questions[4].id, questions[1].id])
    assert kept.questions == (questions[1].id, questions[4].id)
    assert kept.values[PIXELS]["acc"].shape == (2, 2)
    with pytest.raises(ConfigurationError):
        restrict(outcomes, [])


def test_gold_sent_counts_the_crop_sent_and_not_the_crops_scored(screened):
    _, questions, paths = screened
    sent = gold_sent(paths, questions, PIXELS)
    assert sent.tolist() == [True, False, False, False, False, False]
    # the stored metric counts question 1 as well, since gold was scored there
    outcomes = load_outcomes(paths, questions)
    assert outcomes.values[PIXELS]["complete"][:, 0].tolist()[:2] == [1, 1]


def test_gold_sent_refuses_traces_that_disagree(screened, tmp_path):
    _, questions, paths = screened
    rows = [
        _record(q, PIXELS, "gen-c", False, sent="other.png", scored=("other.png",))
        for q in questions
    ]
    other = _write_trace(tmp_path / "trace_other.jsonl", rows)
    with pytest.raises(ConfigurationError, match="disagree"):
        gold_sent([paths[0], other], questions, PIXELS)


def test_closed_book_recoverable_needs_one_generator_only(screened):
    _, questions, paths = screened
    outcomes = load_outcomes(paths, questions)
    assert closed_book_recoverable(outcomes) == {questions[0].id}


def test_text_recoverable_reads_either_corpus_flag(tmp_path):
    path = tmp_path / "integrity.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, ["question_id", "corpus_exact", "corpus_numeric"]
        )
        writer.writeheader()
        writer.writerow(
            {"question_id": "a", "corpus_exact": "True", "corpus_numeric": "False"}
        )
        writer.writerow(
            {"question_id": "b", "corpus_exact": "False", "corpus_numeric": "True"}
        )
        writer.writerow(
            {"question_id": "c", "corpus_exact": "False", "corpus_numeric": "False"}
        )
    assert text_recoverable(path) == {"a", "b"}


def test_gain_difference_is_first_minus_second_and_detects_a_shift():
    shifted = gain_difference([1.0] * 30, [0.0] * 30, iterations=2000, seed=1)
    assert shifted["delta"] == pytest.approx(1.0)
    assert shifted["lower"] == shifted["upper"] == pytest.approx(1.0)
    assert shifted["p"] < 0.01
    same = gain_difference([0.5] * 30, [0.5] * 30, iterations=500, seed=1)
    assert same["delta"] == 0 and same["p"] == 1.0


def test_filtering_change_is_the_kept_mean_minus_the_whole_mean():
    values = [1.0, 1.0, 0.0, 0.0]
    change = filtering_change(
        values, [True, True, False, False], iterations=500, seed=2
    )
    assert change["delta"] == pytest.approx(0.5)
    assert change["lower"] <= change["delta"] <= change["upper"]
    with pytest.raises(ValueError):
        filtering_change(values, [True] * 4, iterations=500)


def test_gains_average_over_generators(screened):
    _, questions, paths = screened
    outcomes = load_outcomes(paths, questions)
    assert np.allclose(gains(outcomes), [1, 1, 1, 0, 0, 0])


def test_cli_writes_the_table_and_one_holm_family(screened, tmp_path, capsys):
    screened_file, questions, paths = screened
    flagged_file = _question_file(tmp_path / "flagged.json", "f", 6)
    flagged_paths = _pixel_traces(
        tmp_path, "f", load_questions(flagged_file), lambda i: i == 2
    )
    obscure_file = _question_file(tmp_path / "obscure.json", "p", 6)
    obscure_paths = _pixel_traces(
        tmp_path, "p", load_questions(obscure_file), lambda i: i == 3
    )
    integrity = tmp_path / "integrity.csv"
    with integrity.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, ["question_id", "corpus_exact", "corpus_numeric"]
        )
        writer.writeheader()
        for i, question in enumerate(questions):
            flag = str(i == 5)
            writer.writerow(
                {
                    "question_id": question.id,
                    "corpus_exact": flag,
                    "corpus_numeric": "False",
                }
            )
    output = tmp_path / "report.json"
    code = main(
        [
            "multimodal-analysis",
            "--screened",
            *map(str, paths),
            "--captions",
            *map(str, _caption_traces(tmp_path, questions)),
            "--flagged",
            *map(str, flagged_paths),
            "--publaynet",
            *map(str, obscure_paths),
            "--screened-questions",
            str(screened_file),
            "--flagged-questions",
            str(flagged_file),
            "--publaynet-questions",
            str(obscure_file),
            "--integrity",
            str(integrity),
            "--iterations",
            "200",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    sizes = {name: column["n"] for name, column in report["columns"].items()}
    assert sizes == {
        "flagged": 6,
        "screened": 6,
        "no_text": 5,
        "no_text_no_closed_book": 4,
        "publaynet": 6,
    }
    screened_arms = report["columns"]["screened"]["arms"]
    assert screened_arms[PIXELS]["acc"] == pytest.approx(0.5)
    assert screened_arms[CAPTIONS]["acc"] == pytest.approx(1 / 12)
    family = report["family"]
    assert len(family) == 8
    assert all(test["p_holm"] >= test["p"] for test in family.values())
    assert family["flagged_minus_screened"]["delta"] == pytest.approx(1 / 6 - 1 / 2)
    assert report["gold_sent"]["screened"]["count"] == 1
    assert report["gold_sent"]["screened"]["accuracy"]["gold_sent"]["n"] == 1
    assert "report ->" in capsys.readouterr().out
