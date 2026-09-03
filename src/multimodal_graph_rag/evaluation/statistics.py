"""Paired inference utilities for the pre-specified evaluation contrasts."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


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
