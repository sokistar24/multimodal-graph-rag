"""Screen visual questions for answer recoverability in captions and corpus text."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from multimodal_graph_rag.evaluation.integrity import audit_answer_recoverability


def _corpus_text(corpus: Path, image_name: str) -> str:
    paper = image_name.split("-Figure", 1)[0].split("-Table", 1)[0]
    candidates = list(corpus.glob(f"{paper}*.txt"))
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in candidates
    )


def verify(
    question_file: Path, captions_file: Path, corpus: Path
) -> list[dict[str, object]]:
    questions = json.loads(question_file.read_text(encoding="utf-8"))
    captions = json.loads(captions_file.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for index, item in enumerate(questions):
        source = str(item["source"])
        caption = str(captions.get(source, ""))
        caption_audit = audit_answer_recoverability(str(item["answer"]), caption)
        corpus_audit = audit_answer_recoverability(
            str(item["answer"]), _corpus_text(corpus, source)
        )
        flagged = caption_audit.needs_human_review or corpus_audit.needs_human_review
        rows.append(
            {
                "question_id": item.get("id", f"figure-{index:04d}"),
                "source": source,
                "construction_method": item.get(
                    "construction_method", "legacy_unspecified"
                ),
                "caption_exact": caption_audit.exact_match,
                "caption_numeric": caption_audit.numeric_equivalent,
                "corpus_exact": corpus_audit.exact_match,
                "corpus_numeric": corpus_audit.numeric_equivalent,
                "audit_status": "flagged_for_human_review"
                if flagged
                else "screened_unflagged",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=Path)
    parser.add_argument("captions", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/runs/figure_integrity.csv")
    )
    args = parser.parse_args()
    rows = verify(args.questions, args.captions, args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(row["audit_status"] == "flagged_for_human_review" for row in rows)
    print(f"wrote {args.output}: {flagged}/{len(rows)} flagged for human review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
