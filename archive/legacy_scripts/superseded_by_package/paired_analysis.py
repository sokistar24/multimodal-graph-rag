"""Compute paired accuracy inference from a detail CSV without extra dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from multimodal_graph_rag.evaluation.statistics import (
    mcnemar_exact,
    paired_bootstrap_delta,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("detail", type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    with args.detail.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    left = [int(row[f"{args.baseline}_acc"]) for row in rows]
    right = [int(row[f"{args.treatment}_acc"]) for row in rows]
    result = {
        "baseline": args.baseline,
        "treatment": args.treatment,
        "bootstrap": paired_bootstrap_delta(
            left, right, iterations=args.iterations, seed=args.seed
        ).__dict__,
        "mcnemar": mcnemar_exact(left, right),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
