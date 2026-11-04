"""The multimodal contrasts of RQ3, pooled over generators.

The primary set is reported four ways: the caption-flagged contrast set, the
screened set, the screened items whose answer is not in the paper's own text,
and those of them no generator answers closed-book. On each, pixels and
captions are compared with text-only retrieval and the oracle image separates
image retrieval from figure reading.

The caption runs hold only the arms that use the image path. Text-only and
closed-book do not use it, so their outcomes come from the pixel run of the
same questions.

``metrics["complete"]`` counts the gold crop among the five crops CLIP scored,
but the pixel arm sends only the top crop and the caption arm the top three, so
what a generator received is read from the evidence items instead.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ConfigurationError
from ..schemas import QuestionRecord
from .pooled import CONTROL_PREFIX, Outcomes, compare
from .posthoc import gold_received, load_trace
from .statistics import holm_adjust

BASELINE = "baseline"
PIXELS = "+multimodal"
CAPTIONS = "+multimodal-captions"
CLOSED_BOOK = f"{CONTROL_PREFIX}closed-book"
ORACLE_IMAGE = f"{CONTROL_PREFIX}oracle-image"
ORACLE_CAPTION = f"{CONTROL_PREFIX}oracle-caption"
CAPTION_ARMS = {PIXELS: CAPTIONS, ORACLE_IMAGE: ORACLE_CAPTION}


def merge_caption_arms(pixels: Outcomes, captions: Outcomes) -> Outcomes:
    """Add a caption run's arms to the pixel run of the same questions."""
    if pixels.questions != captions.questions:
        raise ConfigurationError("the pixel and caption runs hold different questions")
    if set(pixels.generators) != set(captions.generators):
        raise ConfigurationError("the pixel and caption runs hold different generators")
    unknown = set(captions.values) - set(CAPTION_ARMS)
    if unknown:
        raise ConfigurationError(f"not caption arms: {sorted(unknown)}")
    order = [captions.generators.index(name) for name in pixels.generators]
    values = dict(pixels.values)
    strata = dict(pixels.strata)
    for system, name in CAPTION_ARMS.items():
        values[name] = {
            measure: array[:, order]
            for measure, array in captions.values[system].items()
        }
        strata[name] = captions.strata[system][:, order]
    return replace(pixels, values=values, strata=strata)


def restrict(outcomes: Outcomes, keep: Iterable[str]) -> Outcomes:
    """The same outcomes over the questions in ``keep``."""
    kept = set(keep)
    rows = [i for i, qid in enumerate(outcomes.questions) if qid in kept]
    if not rows:
        raise ConfigurationError("no question is left after filtering")
    return Outcomes(
        questions=tuple(outcomes.questions[i] for i in rows),
        generators=outcomes.generators,
        values={
            system: {measure: array[rows] for measure, array in arrays.items()}
            for system, arrays in outcomes.values.items()
        },
        strata={system: array[rows] for system, array in outcomes.strata.items()},
    )


def text_recoverable(integrity_csv: str | Path) -> set[str]:
    """Questions the figure audit flags against the paper's own text."""
    with Path(integrity_csv).open(newline="", encoding="utf-8") as stream:
        return {
            row["question_id"]
            for row in csv.DictReader(stream)
            if row["corpus_exact"] == "True" or row["corpus_numeric"] == "True"
        }


def closed_book_recoverable(outcomes: Outcomes) -> set[str]:
    """Questions at least one generator answers correctly with no evidence."""
    correct = outcomes.values[CLOSED_BOOK]["acc"].max(axis=1) > 0
    return {qid for qid, hit in zip(outcomes.questions, correct, strict=True) if hit}


def gains(outcomes: Outcomes, treatment: str = PIXELS) -> np.ndarray:
    """Per-question accuracy difference from text-only, averaged over generators."""
    arms = outcomes.values
    return (arms[treatment]["acc"] - arms[BASELINE]["acc"]).mean(axis=1)


def gain_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 20261022,
) -> dict[str, float]:
    """Compare two disjoint question sets' mean gains, first minus second.

    The interval resamples each set's questions separately; the p-value permutes
    the set labels, the exchange that no difference between the sets allows.
    """
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both question sets must be non-empty")
    rng = np.random.default_rng(seed)
    draws_a = a[rng.integers(0, a.size, size=(iterations, a.size))].mean(axis=1)
    draws_b = b[rng.integers(0, b.size, size=(iterations, b.size))].mean(axis=1)
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    shuffled = rng.permuted(np.tile(pooled, (iterations, 1)), axis=1)
    permuted = shuffled[:, : a.size].mean(axis=1) - shuffled[:, a.size :].mean(axis=1)
    p = (1 + np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-12)) / (
        iterations + 1
    )
    return {
        "delta": observed,
        "lower": float(np.quantile(draws_a - draws_b, 0.025)),
        "upper": float(np.quantile(draws_a - draws_b, 0.975)),
        "p": float(p),
    }


def filtering_change(
    per_question: Sequence[float],
    kept: Sequence[bool],
    *,
    iterations: int = 10_000,
    seed: int = 20261022,
) -> dict[str, float]:
    """How the mean gain moves when a set is filtered down to ``kept``.

    The interval resamples the whole set, so the filtered set's size varies as it
    did in the sampling. The p-value compares the kept and removed questions.
    """
    values, mask = np.asarray(per_question, float), np.asarray(kept, bool)
    if not mask.any() or mask.all():
        raise ValueError("the filter must keep some questions and remove some")
    draws = np.random.default_rng(seed).integers(
        0, values.size, size=(iterations, values.size)
    )
    sampled, sampled_mask = values[draws], mask[draws]
    counts = sampled_mask.sum(axis=1)
    usable = counts > 0
    filtered = (sampled * sampled_mask).sum(axis=1)[usable] / counts[usable]
    change = filtered - sampled.mean(axis=1)[usable]
    test = gain_difference(
        values[mask], values[~mask], iterations=iterations, seed=seed
    )
    return {
        "delta": float(values[mask].mean() - values.mean()),
        "lower": float(np.quantile(change, 0.025)),
        "upper": float(np.quantile(change, 0.975)),
        "p": test["p"],
    }


def gold_sent(
    traces: Iterable[str | Path], questions: Sequence[QuestionRecord], system: str
) -> np.ndarray:
    """Whether each question's gold crop reached the generator in ``system``.

    Image retrieval does not depend on the generator, so traces that disagree
    point at mixed runs and are refused.
    """
    by_id = {question.id: question for question in questions}
    sent: dict[str, bool] = {}
    for trace in traces:
        for record in load_trace(trace):
            qid = record["question_id"]
            if record["system"] != system or qid not in by_id:
                continue
            question = by_id[qid]
            received = gold_received(record, question) == len(question.gold_sources)
            if sent.setdefault(qid, received) != received:
                raise ConfigurationError(
                    f"traces disagree on the crop sent for question {qid}"
                )
    missing = [question.id for question in questions if question.id not in sent]
    if missing:
        raise ConfigurationError(f"{len(missing)} questions have no {system} record")
    return np.array([sent[question.id] for question in questions])


def by_gold_sent(outcomes: Outcomes, sent: np.ndarray) -> dict[str, dict[str, Any]]:
    """Arm accuracy on the questions whose top crop was gold, and on the others.

    The split follows retrieval, so it is descriptive: questions CLIP ranks well
    can differ from the rest in other ways.
    """
    groups = {"gold_sent": sent, "other_sent": ~sent}
    result: dict[str, dict[str, Any]] = {}
    for name, mask in groups.items():
        result[name] = {"n": int(mask.sum())}
        if mask.any():
            for system, arrays in outcomes.values.items():
                result[name][system] = float(arrays["acc"][mask].mean())
    return result


def _column(outcomes: Outcomes, iterations: int, seed: int) -> dict[str, Any]:
    report = compare(outcomes, BASELINE, iterations=iterations, seed=seed)
    return {
        "n": report["n_questions"],
        "generators": report["generators"],
        "arms": {
            system: {
                "acc": entry["acc"],
                **(
                    {"difference": entry["differences"]["acc"]}
                    if "acc" in entry["differences"]
                    else {}
                ),
            }
            for system, entry in report["systems"].items()
        },
        "per_generator": {
            system: {name: cells["acc"] for name, cells in generators.items()}
            for system, generators in report["per_generator"].items()
        },
    }


def analyse(
    *,
    flagged: Outcomes,
    screened: Outcomes,
    obscure: Outcomes,
    recoverable_from_text: set[str],
    flagged_sent: np.ndarray,
    screened_sent: np.ndarray,
    obscure_sent: np.ndarray,
    iterations: int = 10_000,
    seed: int = 20261022,
) -> dict[str, Any]:
    """Every RQ3 number, with one Holm family over the tests of H3 and PubLayNet."""
    if set(flagged.questions) & set(screened.questions):
        raise ConfigurationError("the flagged and screened sets share questions")
    without_text = [q for q in screened.questions if q not in recoverable_from_text]
    from_closed_book = closed_book_recoverable(screened)
    without_both = [q for q in without_text if q not in from_closed_book]
    columns = {
        "flagged": _column(flagged, iterations, seed),
        "screened": _column(screened, iterations, seed),
        "no_text": _column(restrict(screened, without_text), iterations, seed),
        "no_text_no_closed_book": _column(
            restrict(screened, without_both), iterations, seed
        ),
        "publaynet": _column(obscure, iterations, seed),
    }
    screened_gains = gains(screened)
    tests: dict[str, dict[str, Any]] = {
        "pixels_flagged": dict(columns["flagged"]["arms"][PIXELS]["difference"]),
        "pixels_screened": dict(columns["screened"]["arms"][PIXELS]["difference"]),
        "pixels_no_text": dict(columns["no_text"]["arms"][PIXELS]["difference"]),
        "pixels_no_text_no_closed_book": dict(
            columns["no_text_no_closed_book"]["arms"][PIXELS]["difference"]
        ),
        "captions_screened": dict(columns["screened"]["arms"][CAPTIONS]["difference"]),
        "pixels_publaynet": dict(columns["publaynet"]["arms"][PIXELS]["difference"]),
        "flagged_minus_screened": gain_difference(
            gains(flagged), screened_gains, iterations=iterations, seed=seed
        ),
        "filtered_minus_screened": filtering_change(
            screened_gains,
            [q in set(without_both) for q in screened.questions],
            iterations=iterations,
            seed=seed,
        ),
    }
    names = list(tests)
    for name, adjusted in zip(
        names, holm_adjust(tests[name]["p"] for name in names), strict=True
    ):
        tests[name]["p_holm"] = adjusted
    return {
        "columns": columns,
        "family": tests,
        "filtered_out": {
            "text_recoverable": sorted(set(screened.questions) & recoverable_from_text),
            "closed_book_recoverable": sorted(from_closed_book),
        },
        "gold_sent": {
            "flagged": _sent_summary(flagged, flagged_sent),
            "screened": _sent_summary(screened, screened_sent),
            "publaynet": _sent_summary(obscure, obscure_sent),
        },
    }


def _sent_summary(outcomes: Outcomes, sent: np.ndarray) -> dict[str, Any]:
    return {
        "rate": float(sent.mean()),
        "count": int(sent.sum()),
        "accuracy": by_gold_sent(outcomes, sent),
    }


__all__ = [
    "BASELINE",
    "CAPTIONS",
    "CLOSED_BOOK",
    "ORACLE_CAPTION",
    "ORACLE_IMAGE",
    "PIXELS",
    "analyse",
    "by_gold_sent",
    "closed_book_recoverable",
    "filtering_change",
    "gain_difference",
    "gains",
    "gold_sent",
    "merge_caption_arms",
    "restrict",
    "text_recoverable",
]
