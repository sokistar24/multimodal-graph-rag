"""
Dumps the ranked top-k text retrieval for every cross-paper question, with hit
flags for BOTH gold papers. Retrieval only — no generation, no judging; cost is
~50 query embeddings.

Run with the SPIQA corpus active:
    $env:RAG_CORPUS = "spiqa"
    python retrieval_dump.py questions_spiqa_multihop_cross.json

Output: retrieval_dump_<questionfile>.json, one record per question:
    {"q", "source", "source2", "top_k": [[score, src], ...],
     "source_in_topk", "source2_in_topk", "both_in_topk"}

This supplies the two numbers the detail CSVs cannot: how often the SECOND gold
paper is retrieved, and how often BOTH are — the completeness ceiling for
cross-document questions, and the target list for the +KG-retrieval arm.
"""

import json
import os
import sys

from multimodal_graph_rag.retrieval.text import CORPUS_DIR, build_index, retrieve

K = 3


def main(qfile):
    print(f"Corpus: {CORPUS_DIR}")
    index, chunks, sources = build_index()
    with open(qfile, encoding="utf-8") as f:
        questions = json.load(f)

    out, s1_hits, s2_hits, both_hits = [], 0, 0, 0
    for item in questions:
        ranked = retrieve(item["q"], index, chunks, sources, k=K)
        top = [[round(score, 4), src] for score, src, _ in ranked]
        top_sources = {src for _, src in top}
        h1 = item["source"] in top_sources
        h2 = item.get("source2") in top_sources
        s1_hits += h1
        s2_hits += h2
        both_hits += h1 and h2
        out.append(
            {
                "q": item["q"],
                "source": item["source"],
                "source2": item.get("source2"),
                "top_k": top,
                "source_in_topk": h1,
                "source2_in_topk": h2,
                "both_in_topk": h1 and h2,
            }
        )

    n = len(out)
    dump = f"retrieval_dump_{os.path.splitext(os.path.basename(qfile))[0]}.json"
    with open(dump, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n{n} questions -> {dump}")
    print(
        f"source  in top-{K}: {s1_hits}/{n}  ({s1_hits / n:.3f})   <- matches summary Recall@3"
    )
    print(f"source2 in top-{K}: {s2_hits}/{n}  ({s2_hits / n:.3f})")
    print(
        f"BOTH    in top-{K}: {both_hits}/{n}  ({both_hits / n:.3f})   <- completeness ceiling"
    )
    print(
        f"\nQuestions missing source2 (the +KG-retrieval arm's rescue targets): "
        f"{n - s2_hits}"
    )


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join("data", "questions", "questions_spiqa_multihop_cross.json")
    )
