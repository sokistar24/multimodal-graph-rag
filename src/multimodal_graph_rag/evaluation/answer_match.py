"""Reference-based answer scoring that needs no model and no human labels.

Exact match and token F1 against the reference answer, using the normalisation
the extractive QA benchmarks use: lowercase, drop punctuation and articles,
collapse whitespace.

These matter because the reference answers on the human-authored sets were
written by the benchmarks' own annotators. Scoring against them is therefore a
*human-derived* correctness signal that involves no LLM judge at all, which is
what lets the graph experiments report a headline number without a calibrated
judge. On HotpotQA, whose answers are short spans, exact match and F1 are the
benchmark's official metrics and should be the primary outcome.

They are reported alongside the judge on every set. Where the two disagree the
gap is itself informative: on discursive answers exact match is far too strict,
so a large judge-minus-exact-match gap marks the sets where the judge is doing
real work and is therefore the load-bearing instrument.
"""

from __future__ import annotations

import re
import string
from collections import Counter

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalise_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, and collapse whitespace."""
    lowered = text.lower().translate(PUNCTUATION)
    return " ".join(ARTICLES.sub(" ", lowered).split())


def answer_tokens(text: str) -> list[str]:
    return normalise_answer(text).split()


def exact_match(prediction: str, reference: str) -> int:
    """1 when the normalised strings are identical, else 0."""
    return int(normalise_answer(prediction) == normalise_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between prediction and reference.

    Two empty answers count as a perfect match, and an empty answer against a
    non-empty one counts as zero, matching the extractive QA convention.
    """
    predicted, expected = answer_tokens(prediction), answer_tokens(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    shared = Counter(predicted) & Counter(expected)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def contains_reference(prediction: str, reference: str) -> int:
    """1 when the normalised reference appears inside the normalised prediction.

    A lenient companion to exact match, for generators that answer in a
    sentence rather than with a bare span. It is reported, never substituted
    for exact match, because it cannot distinguish a correct answer from one
    that also asserts something false.
    """
    expected = normalise_answer(reference)
    return int(bool(expected) and expected in normalise_answer(prediction))
