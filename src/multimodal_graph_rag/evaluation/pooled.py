"""Comparisons pooled over generators, with the question as the unit of analysis.

Each generator's trace of one question set gives an outcome per question and
system. Every system is compared with a reference system on the same
questions: the difference is the mean over question-by-generator cells, its
interval comes from a question-level cluster bootstrap, and its p-value from a
sign-flip test on the per-question mean differences, Holm-adjusted within each
measure over the systems that are not evidence controls.

Exact match, F1, and containment are recomputed from the stored answers with
the current scoring, so a change in scoring reaches every reported number;
judge accuracy, faithfulness, and retrieval come from the stored records. For
each named pair of systems the access and use terms of the accuracy
difference are reported with intervals from the same resampling.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ConfigurationError
from ..schemas import QuestionRecord
from .answer_match import contains_reference, exact_match, token_f1
from .statistics import (
    access_use_split,
    cluster_bootstrap_delta,
    holm_adjust,
    sign_flip_test,
)

MEASURES = ("em", "f1", "contains", "acc")
BINARY_MEASURES = ("em", "contains", "acc")
CONTROL_PREFIX = "control:"


@dataclass(frozen=True)
class Outcomes:
    """Outcomes of every system, as question-by-generator arrays."""

    questions: tuple[str, ...]
    generators: tuple[str, ...]
    values: Mapping[str, Mapping[str, np.ndarray]]
    strata: Mapping[str, np.ndarray]

    @property
    def systems(self) -> list[str]:
        return list(self.values)


def _stratum(record: Mapping[str, Any]) -> str:
    metrics = record["metrics"]
    if int(metrics["complete"]):
        return "complete"
    return "partial" if float(metrics["recall"]) > 0 else "none"


def load_outcomes(
    traces: Iterable[str | Path], questions: Sequence[QuestionRecord]
) -> Outcomes:
    """Read one trace per generator into arrays over questions and generators."""
    by_id = {question.id: question for question in questions}
    records: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    generators: list[str] = []
    for trace in traces:
        with Path(trace).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        names = {name for row in rows for name in row["model_versions"]}
        if len(names) != 1:
            raise ConfigurationError(f"{trace} holds answers of {sorted(names)}")
        generator = names.pop()
        if generator in generators:
            raise ConfigurationError(f"two traces for {generator}")
        generators.append(generator)
        for row in rows:
            if row["question_id"] in by_id:
                records[(row["question_id"], row["system"], generator)] = row
    if not generators:
        raise ConfigurationError("no traces given")
    systems = sorted({system for _, system, _ in records})
    ids = tuple(q.id for q in questions)
    missing = [
        (qid, system, generator)
        for qid in ids
        for system in systems
        for generator in generators
        if (qid, system, generator) not in records
    ]
    if missing:
        raise ConfigurationError(
            f"{len(missing)} question-system-generator outcomes are missing, "
            f"e.g. {missing[:3]}"
        )
    shape = (len(ids), len(generators))
    values: dict[str, dict[str, np.ndarray]] = {}
    strata: dict[str, np.ndarray] = {}
    for system in systems:
        arrays = {measure: np.zeros(shape) for measure in (*MEASURES, "faith")}
        arrays["complete"] = np.zeros(shape)
        stratum = np.empty(shape, dtype=object)
        for i, qid in enumerate(ids):
            reference = by_id[qid].answer
            for j, generator in enumerate(generators):
                row = records[(qid, system, generator)]
                answer = row["answer"]
                arrays["em"][i, j] = exact_match(answer, reference)
                arrays["f1"][i, j] = token_f1(answer, reference)
                arrays["contains"][i, j] = contains_reference(answer, reference)
                arrays["acc"][i, j] = float(row["metrics"]["acc"])
                faith = row["metrics"]["faith"]
                arrays["faith"][i, j] = np.nan if faith is None else float(faith)
                arrays["complete"][i, j] = float(row["metrics"]["complete"])
                stratum[i, j] = _stratum(row)
        values[system] = arrays
        strata[system] = stratum
    return Outcomes(ids, tuple(generators), values, strata)


def _split_interval(
    outcomes: Outcomes,
    treatment: str,
    reference: str,
    measure: str,
    draws: np.ndarray,
) -> dict[str, list[float]]:
    def split(rows: np.ndarray | slice) -> tuple[float, ...]:
        result = access_use_split(
            outcomes.values[reference][measure][rows].ravel(),
            outcomes.strata[reference][rows].ravel(),
            outcomes.values[treatment][measure][rows].ravel(),
            outcomes.strata[treatment][rows].ravel(),
        )
        return (
            result.delta,
            result.access,
            result.use,
            result.symmetric_access,
            result.symmetric_use,
        )

    point = split(slice(None))
    samples = np.array([split(rows) for rows in draws])
    names = ("delta", "access", "use", "symmetric_access", "symmetric_use")
    return {
        name: [
            float(point[k]),
            float(np.quantile(samples[:, k], 0.025)),
            float(np.quantile(samples[:, k], 0.975)),
        ]
        for k, name in enumerate(names)
    }


def compare(
    outcomes: Outcomes,
    reference: str,
    *,
    splits: Sequence[tuple[str, str]] = (),
    iterations: int = 10_000,
    seed: int = 20261022,
) -> dict[str, Any]:
    """Every system against ``reference``, pooled over generators."""
    if reference not in outcomes.values:
        raise ConfigurationError(f"reference system {reference!r} is not in the traces")
    for treatment, base in splits:
        for system in (treatment, base):
            if system not in outcomes.values:
                raise ConfigurationError(f"system {system!r} is not in the traces")
    report: dict[str, Any] = {
        "reference": reference,
        "generators": list(outcomes.generators),
        "n_questions": len(outcomes.questions),
        "systems": {},
        "per_generator": {},
        "splits": {},
    }
    tested: dict[str, list[tuple[str, float]]] = {m: [] for m in BINARY_MEASURES}
    for system in outcomes.systems:
        arrays = outcomes.values[system]
        entry: dict[str, Any] = {
            "complete": float(arrays["complete"].mean()),
            **{measure: float(arrays[measure].mean()) for measure in MEASURES},
            "differences": {},
        }
        report["per_generator"][system] = {
            generator: {m: float(arrays[m][:, j].mean()) for m in MEASURES}
            for j, generator in enumerate(outcomes.generators)
        }
        if system != reference:
            base = outcomes.values[reference]
            for measure in MEASURES:
                estimate = cluster_bootstrap_delta(
                    base[measure], arrays[measure], iterations=iterations, seed=seed
                )
                difference: dict[str, Any] = {
                    "delta": estimate.delta,
                    "lower": estimate.lower,
                    "upper": estimate.upper,
                }
                if measure in BINARY_MEASURES:
                    difference["wins"] = int(
                        np.count_nonzero((arrays[measure] == 1) & (base[measure] == 0))
                    )
                    difference["losses"] = int(
                        np.count_nonzero((arrays[measure] == 0) & (base[measure] == 1))
                    )
                    difference["p"] = sign_flip_test(
                        (arrays[measure] - base[measure]).mean(axis=1),
                        iterations=iterations,
                        seed=seed,
                    )
                    if not system.startswith(CONTROL_PREFIX):
                        tested[measure].append((system, difference["p"]))
                entry["differences"][measure] = difference
        report["systems"][system] = entry
    for measure, results in tested.items():
        for (system, _), adjusted in zip(
            results, holm_adjust(p for _, p in results), strict=True
        ):
            report["systems"][system]["differences"][measure]["p_holm"] = adjusted
    draws = np.random.default_rng(seed).integers(
        0, len(outcomes.questions), size=(iterations, len(outcomes.questions))
    )
    for treatment, base in splits:
        report["splits"][f"{treatment} vs {base}"] = {
            measure: _split_interval(outcomes, treatment, base, measure, draws)
            for measure in BINARY_MEASURES
        }
    return report


def adjust_family(reports: Sequence[dict[str, Any]]) -> None:
    """Holm-adjust the tests of several reports as one family, per measure.

    A pre-specified family can span question sets, such as the bridge and
    comparison halves of one research question; each report's ``p_holm`` is
    replaced by the value adjusted over every non-control test in the family.
    """
    for measure in BINARY_MEASURES:
        tests = [
            entry["differences"][measure]
            for report in reports
            for system, entry in report["systems"].items()
            if not system.startswith(CONTROL_PREFIX) and measure in entry["differences"]
        ]
        for test, adjusted in zip(
            tests, holm_adjust(test["p"] for test in tests), strict=True
        ):
            test["p_holm"] = adjusted


__all__ = ["MEASURES", "Outcomes", "adjust_family", "compare", "load_outcomes"]
