"""Index our corpus with HippoRAG 2 and freeze its rankings to a retrievals file.

Run this in the comparators environment (``pip install -e .[comparators]``),
never in an evaluation:

    python -m multimodal_graph_rag.retrieval.hipporag_index \
        --corpus-dir hotpotqa_corpus \
        --questions data/questions/questions_hotpotqa_bridge.json \
        --output .cache/hipporag/hotpotqa/retrievals_hotpotqa_bridge.json \
        --save-dir .cache/hipporag/hotpotqa/index

The output is what ``comparator_retrievals`` in an experiment config points
at, and what the ``+HippoRAG`` system serves. Indexing is LLM-based and not
deterministic across rebuilds, so build once per corpus and keep the file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from importlib import metadata

from .comparators import RetrievalEngine, precompute_hipporag

EngineFactory = Callable[..., RetrievalEngine]


def _hipporag_factory(
    *, save_dir: str, llm_model_name: str, embedding_model_name: str
) -> RetrievalEngine:
    try:
        # Deliberately lazy: the main environment never has hipporag installed.
        from hipporag import HippoRAG  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "hipporag is not installed in this environment; run "
            "`pip install -e .[comparators]` in the comparators environment"
        ) from exc
    return HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
    )


def _package_version() -> str:
    try:
        return metadata.version("hipporag")
    except metadata.PackageNotFoundError:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hipporag_index",
        description="freeze HippoRAG 2 rankings over our corpus chunks",
    )
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-dir", required=True, help="HippoRAG index directory")
    parser.add_argument("--llm", default="gpt-4o-mini", help="OpenIE model")
    parser.add_argument("--embedding", default="text-embedding-3-small")
    parser.add_argument(
        "--num-to-retrieve",
        type=int,
        default=10,
        help="rankings kept per question; the evaluation truncates to its budget",
    )
    return parser


def main(
    argv: Sequence[str] | None = None, *, engine_factory: EngineFactory | None = None
) -> int:
    args = build_parser().parse_args(argv)
    factory = engine_factory or _hipporag_factory
    engine = factory(
        save_dir=args.save_dir,
        llm_model_name=args.llm,
        embedding_model_name=args.embedding,
    )
    output = precompute_hipporag(
        engine,
        corpus_dir=args.corpus_dir,
        question_file=args.questions,
        output=args.output,
        num_to_retrieve=args.num_to_retrieve,
        llm_model=args.llm,
        embedding_model=args.embedding,
        package_version=_package_version(),
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
