"""Paired inference utilities for the pre-specified evaluation contrasts."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedEstimate:
    delta: float
    lower: float
    upper: float
    confidence: float
    samples: int


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_delta(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedEstimate:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired samples must be non-empty and equally sized")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    deltas = [float(t) - float(b) for b, t in zip(baseline, treatment, strict=True)]
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        draws.append(
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    alpha = 1 - confidence
    return PairedEstimate(
        delta=sum(deltas) / len(deltas),
        lower=_quantile(draws, alpha / 2),
        upper=_quantile(draws, 1 - alpha / 2),
        confidence=confidence,
        samples=len(deltas),
    )


def mcnemar_exact(
    baseline: Sequence[int], treatment: Sequence[int]
) -> dict[str, float | int]:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired samples must be non-empty and equally sized")
    b = sum(int(x == 1 and y == 0) for x, y in zip(baseline, treatment, strict=True))
    c = sum(int(x == 0 and y == 1) for x, y in zip(baseline, treatment, strict=True))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, i) for i in range(min(b, c) + 1)) / (
            2**discordant
        )
        p_value = min(1.0, 2 * tail)
    odds_ratio = (c + 0.5) / (b + 0.5)
    return {
        "baseline_only": b,
        "treatment_only": c,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def _question_matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("expected one row of outcomes per question")
    return matrix


def cluster_bootstrap_delta(
    reference: Sequence[Sequence[float]],
    treatment: Sequence[Sequence[float]],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedEstimate:
    """Mean paired difference over question-by-generator cells.

    Rows are questions and columns generators. Questions are resampled with
    all their generator outcomes together, so answers to one question are
    never treated as independent.
    """
    before, after = _question_matrix(reference), _question_matrix(treatment)
    if before.shape != after.shape:
        raise ValueError("reference and treatment must have the same shape")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    per_question = (after - before).mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_question), size=(iterations, len(per_question)))
    means = per_question[draws].mean(axis=1)
    alpha = 1 - confidence
    return PairedEstimate(
        delta=float(per_question.mean()),
        lower=float(np.quantile(means, alpha / 2)),
        upper=float(np.quantile(means, 1 - alpha / 2)),
        confidence=confidence,
        samples=len(per_question),
    )


def sign_flip_test(
    differences: Sequence[float], *, iterations: int = 10_000, seed: int = 0
) -> float:
    """Two-sided permutation p-value for a zero mean paired difference.

    Each question's difference keeps its size and takes a random sign, which
    is the exchange the null hypothesis of no system effect allows.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("expected one difference per question")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    observed = abs(values.mean())
    signs = np.random.default_rng(seed).choice(
        [-1.0, 1.0], size=(iterations, values.size)
    )
    permuted = np.abs((signs * values).mean(axis=1))
    return float(
        (1 + np.count_nonzero(permuted >= observed - 1e-12)) / (iterations + 1)
    )


@dataclass(frozen=True)
class AccessUse:
    delta: float
    access: float
    use: float
    symmetric_access: float
    symmetric_use: float


def access_use_split(
    reference_correct: Sequence[int],
    reference_strata: Sequence[str],
    treatment_correct: Sequence[int],
    treatment_strata: Sequence[str],
) -> AccessUse:
    """Split a difference in accuracy into access and use terms.

    With stratum shares p and per-stratum accuracies a, the difference is
    sum (p_t - p_r) a_r + sum p_t (a_t - a_r): the change expected from moving
    answers between completeness strata at the reference's accuracy, plus the
    change in accuracy within strata. The symmetric terms average this split
    with the one that takes the treatment as the reference (Kitagawa, 1955).
    A stratum one system never reaches has accuracy 0 for that system.
    """
    if not (
        len(reference_correct)
        == len(reference_strata)
        == len(treatment_correct)
        == len(treatment_strata)
        > 0
    ):
        raise ValueError("all four sequences must be non-empty and equally long")
    y_r, y_t = (
        np.asarray(reference_correct, float),
        np.asarray(treatment_correct, float),
    )
    k_r, k_t = np.asarray(reference_strata), np.asarray(treatment_strata)
    access = use = symmetric_access = symmetric_use = 0.0
    for stratum in sorted(set(k_r) | set(k_t)):
        in_r, in_t = k_r == stratum, k_t == stratum
        p_r, p_t = in_r.mean(), in_t.mean()
        a_r = y_r[in_r].mean() if in_r.any() else 0.0
        a_t = y_t[in_t].mean() if in_t.any() else 0.0
        access += (p_t - p_r) * a_r
        use += p_t * (a_t - a_r)
        symmetric_access += (p_t - p_r) * (a_t + a_r) / 2
        symmetric_use += (p_t + p_r) / 2 * (a_t - a_r)
    return AccessUse(
        delta=float(y_t.mean() - y_r.mean()),
        access=float(access),
        use=float(use),
        symmetric_access=float(symmetric_access),
        symmetric_use=float(symmetric_use),
    )


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, (original_index, value) in enumerate(indexed):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[original_index] = running
    return adjusted
