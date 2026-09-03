"""Convert legacy question JSON into the versioned QuestionRecord schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_graph_rag.schemas import QuestionRecord, content_hash


def convert(source: Path, destination: Path) -> str:
    with source.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    records = [
        QuestionRecord.from_legacy(item, index).to_dict()
        for index, item in enumerate(raw)
    ]
    payload = {
        "schema_version": 1,
        "source_file": source.name,
        "question_count": len(records),
        "questions_sha256": content_hash(records),
        "questions": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return payload["questions_sha256"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    digest = convert(args.source, args.destination)
    print(f"wrote {args.destination} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
