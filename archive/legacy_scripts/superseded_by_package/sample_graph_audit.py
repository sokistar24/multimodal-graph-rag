"""Create a deterministic, corpus-stratified triple-audit sheet."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

AUDIT_FIELDS = [
    "audit_id",
    "corpus",
    "cache_key",
    "subject",
    "relation",
    "object",
    "extraction_frequency",
    "triple_correct",
    "source_entails",
    "subject_canonical",
    "object_canonical",
    "relation_specific",
    "provenance_correct",
    "reviewer",
    "notes",
]


def sample(cache: Path, corpus: str, count: int, seed: int) -> list[dict[str, object]]:
    raw = json.loads(cache.read_text(encoding="utf-8"))
    occurrences: dict[tuple[str, str, str], list[str]] = {}
    for key, triples in raw.items():
        for triple in triples:
            if len(triple) >= 3:
                value = tuple(str(part) for part in triple[:3])
                occurrences.setdefault(value, []).append(key)
    ranked = sorted(occurrences, key=lambda triple: (-len(occurrences[triple]), triple))
    if count > len(ranked):
        raise ValueError(
            f"requested {count} triples but cache contains {len(ranked)} unique triples"
        )
    # Half frequency-stratified, half uniform random, without duplicates.
    bins = [ranked[i::4] for i in range(4)]
    rng = random.Random(seed)
    chosen: list[tuple[str, str, str]] = []
    per_bin = count // 4
    for group in bins:
        chosen.extend(rng.sample(group, min(per_bin, len(group))))
    remaining = [triple for triple in ranked if triple not in set(chosen)]
    chosen.extend(rng.sample(remaining, count - len(chosen)))
    rows = []
    for index, triple in enumerate(chosen):
        subject, relation, obj = triple
        rows.append(
            {
                "audit_id": f"{corpus}-{index + 1:04d}",
                "corpus": corpus,
                "cache_key": occurrences[triple][0],
                "subject": subject,
                "relation": relation,
                "object": obj,
                "extraction_frequency": len(occurrences[triple]),
                "triple_correct": "",
                "source_entails": "",
                "subject_canonical": "",
                "object_canonical": "",
                "relation_specific": "",
                "provenance_correct": "",
                "reviewer": "",
                "notes": "",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output", type=Path, default=Path("protocols/graph_triple_audit_sample.csv")
    )
    args = parser.parse_args()
    rows = sample(args.cache, args.corpus, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} triples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
