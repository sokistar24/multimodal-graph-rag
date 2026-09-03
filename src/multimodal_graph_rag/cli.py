"""The ``rag`` command: ingestion, question authoring, evaluation, and artifacts.

Every command that spends provider credit builds one :class:`ModelClient` from
an explicit pricing snapshot; secrets are read from the environment (and a
``.env`` file, when present) only when a command starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from .artifacts import (
    build_figures,
    build_tables,
    freeze_manifest,
    load_frozen_summary_rows,
    scan_runs,
    select_from_manifest,
    validate_manifest,
    write_matrix,
)
from .clients import GENERATORS, ModelClient, load_pricing
from .config import ExperimentConfig, load_experiment_config
from .corpora import corpus_spec, docbank, doclaynet, hotpotqa, publaynet, spiqa
from .errors import ConfigurationError, MultimodalGraphRagError
from .evaluation.audits import (
    FIGURE_AUDIT_FIELDS,
    TRIPLE_AUDIT_FIELDS,
    figure_integrity_rows,
    paired_accuracy,
    sample_triple_audit,
)
from .evaluation.statistics import mcnemar_exact, paired_bootstrap_delta
from .pipelines.evaluate import EvaluationSettings, run_evaluation
from .questions import clean_question_file
from .questions.authoring import author_figure_questions, author_text_questions
from .questions.caption_matched import author_caption_matched
from .questions.cross_paper import author_cross_paper
from .retrieval.graph import default_cache_path, load_graph
from .retrieval.graph_expansion import bridge_report
from .retrieval.text import TextIndex
from .schemas import load_questions

QUESTIONS_DIR = Path("data/questions")
GRAPHS_DIR = Path("data/graphs")
SPIQA_DOWNLOAD_DIR = Path("spiqa_download")


def _client(pricing_path: str | Path) -> ModelClient:
    return ModelClient(load_pricing(pricing_path))


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ConfigurationError(
            "not inside a git repository; pass --commit explicitly"
        )
    return result.stdout.strip()


def cmd_ingest(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    spec = corpus_spec(config.corpus)
    client = None if args.smoke else _client(config.pricing_snapshot)
    image_dir = config.image_dir or (
        str(spec.default_image_dir) if spec.default_image_dir else None
    )
    if config.corpus == "publaynet":
        report = publaynet.ingest(
            corpus_dir=config.corpus_dir,
            image_dir=image_dir,
            client=client,
            limit=args.limit,
            shards=args.shards,
            smoke=args.smoke,
        )
    elif config.corpus == "spiqa":
        report = spiqa.ingest(
            corpus_dir=config.corpus_dir,
            image_dir=image_dir,
            download_dir=SPIQA_DOWNLOAD_DIR,
            gold_qa_path=QUESTIONS_DIR / "spiqa_gold_qa.json",
            client=client,
            papers=args.papers,
            seed=config.seed or spiqa.DEFAULT_SEED,
            smoke=args.smoke,
        )
    elif config.corpus == "hotpotqa":
        report = hotpotqa.ingest(
            corpus_dir=config.corpus_dir,
            questions_dir=QUESTIONS_DIR,
            pool=args.pool,
            evaluate=args.eval,
            seed=config.seed or hotpotqa.DEFAULT_SEED,
            clean=args.clean,
            smoke=args.smoke,
        )
    elif config.corpus == "docbank":
        report = docbank.ingest(
            corpus_dir=config.corpus_dir,
            image_dir=image_dir,
            client=client,
            limit=args.limit,
            smoke=args.smoke,
        )
    else:
        report = doclaynet.ingest(
            corpus_dir=config.corpus_dir,
            image_dir=image_dir,
            client=client,
            limit=args.limit,
            category_filter=args.category,
            smoke=args.smoke,
        )
    print(json.dumps(report.__dict__, indent=2))
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    config: ExperimentConfig | None = (
        load_experiment_config(args.config) if args.config else None
    )
    corpus_dir = args.corpus_dir or (config.corpus_dir if config else None)
    image_dir = args.image_dir or (config.image_dir if config else None)
    pricing = args.pricing or (config.pricing_snapshot if config else None)
    if not pricing:
        raise ConfigurationError("--pricing or --config is required")
    client = _client(pricing)
    if args.protocol in ("text", "multihop"):
        if not corpus_dir or not args.out:
            raise ConfigurationError(
                "text/multihop authoring needs --corpus-dir and --out"
            )
        report = author_text_questions(
            client,
            corpus_dir=corpus_dir,
            kind=args.protocol,
            target=args.target,
            out_path=args.out,
            seed=args.seed,
            smoke=args.smoke,
        )
    elif args.protocol == "figure":
        if not image_dir or not args.out:
            raise ConfigurationError("figure authoring needs --image-dir and --out")
        report = author_figure_questions(
            client,
            image_dir=image_dir,
            target=args.target,
            out_path=args.out,
            seed=args.seed,
            smoke=args.smoke,
        )
    elif args.protocol == "caption-matched":
        if not args.pixel_file or not image_dir or not args.out:
            raise ConfigurationError(
                "caption-matched authoring needs --pixel-file, --image-dir, and --out"
            )
        report = author_caption_matched(
            client,
            pixel_file=args.pixel_file,
            captions_file=Path(image_dir) / "captions.json",
            out_path=args.out,
        )
    else:
        if not corpus_dir or not args.out:
            raise ConfigurationError(
                "cross-paper authoring needs --corpus-dir and --out"
            )
        report = author_cross_paper(
            client,
            corpus_dir=corpus_dir,
            cache_path=args.graph_cache or default_cache_path(corpus_dir, GRAPHS_DIR),
            target=args.target,
            out_path=args.out,
            seed=args.seed,
            verbose=args.verbose,
        )
    print(
        json.dumps(
            {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in report.__dict__.items()
            },
            indent=2,
        )
    )
    print("\nEvery authored item still needs manual review before it is used.")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    missing = config.validate_paths()
    if missing and not args.dry_run:
        raise ConfigurationError("missing configured paths:\n  " + "\n  ".join(missing))
    models = [args.model] if args.model else list(config.generators)
    unknown = [m for m in models if m not in GENERATORS and m != "gpt4o"]
    if unknown:
        raise ConfigurationError(f"unknown generators in config: {unknown}")
    if args.dry_run:
        for model in models:
            settings = EvaluationSettings.from_config(
                config, model, pricing_captured_on="dry-run", limit=args.limit
            )
            print(
                f"{model}: systems={list(settings.systems)} questions={settings.question_file} corpus={settings.corpus_dir}"
            )
        print(f"config identity: {config.config_hash}")
        return 0
    pricing = load_pricing(config.pricing_snapshot)
    client = ModelClient(pricing)
    for model in models:
        settings = EvaluationSettings.from_config(
            config, model, pricing_captured_on=pricing.captured_on, limit=args.limit
        )
        run_evaluation(settings, client)
    return 0


def cmd_retrieval_ab(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    client = _client(config.pricing_snapshot)
    questions = load_questions(config.question_set)
    index = TextIndex.build(config.corpus_dir, client, cache_dir=config.cache_dir)
    graph = load_graph(
        config.corpus_dir, default_cache_path(config.corpus_dir, GRAPHS_DIR)
    )
    hops = args.hops or config.bridge_hops
    report = bridge_report(
        questions,
        index,
        graph,
        client,
        k=min(3, config.candidate_budget),
        n_bridge=config.candidate_budget - min(3, config.candidate_budget),
        hops=hops,
    )
    output = Path(
        args.output
        or f"artifacts/runs/retrieval_ab_{Path(config.question_set).stem}_h{hops}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "summary": report.summary(),
                "outcomes": [o.to_dict() for o in report.outcomes],
            },
            stream,
            indent=2,
        )
    print(json.dumps(report.summary(), indent=2))
    print(f"wrote {output}")
    return 0


def cmd_artifacts(args: argparse.Namespace) -> int:
    if args.manifest:
        errors = validate_manifest(args.manifest)
        if errors:
            raise ConfigurationError("manifest is invalid:\n  " + "\n  ".join(errors))
        selection = select_from_manifest(args.manifest)
        rows = load_frozen_summary_rows(args.manifest)
    else:
        selection = scan_runs(args.scan)
        rows = selection.summary_rows()
        print("exploratory selection from a directory scan; not a release artifact")
    out_dir = Path(args.out)
    write_matrix(rows, out_dir / "tables" / "matrix_all.csv")
    tables = build_tables(
        selection,
        out_dir / "tables",
        spiqa_figure_questions=args.spiqa_figures,
        spiqa_corpus=args.spiqa_corpus,
    )
    figures = build_figures(selection, out_dir / "tables", out_dir / "figures")
    for qset, models in sorted(selection.coverage().items()):
        print(
            f"  {qset:<32} generators: {len(models & set(GENERATORS))}/{len(GENERATORS)}"
        )
    for path in tables + figures:
        print(f"wrote {path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    errors = validate_manifest(args.manifest)
    if errors:
        print("release validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("release manifest and checksums are valid")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    manifest = freeze_manifest(
        args.summary,
        release_id=args.release_id,
        commit=args.commit or _git_commit(),
        question_set=args.question_set,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"wrote {output} ({len(manifest['runs'])} runs)")
    return 0


def cmd_audit_graph(args: argparse.Namespace) -> int:
    rows = sample_triple_audit(args.cache, args.corpus, args.count, args.seed)
    _write_csv(Path(args.output), TRIPLE_AUDIT_FIELDS, rows)
    print(f"wrote {args.output} ({len(rows)} triples)")
    return 0


def cmd_audit_figures(args: argparse.Namespace) -> int:
    rows = figure_integrity_rows(args.questions, args.captions, args.corpus)
    _write_csv(Path(args.output), FIGURE_AUDIT_FIELDS, rows)
    flagged = sum(row["audit_status"] == "flagged_for_human_review" for row in rows)
    print(f"wrote {args.output}: {flagged}/{len(rows)} flagged for human review")
    return 0


def cmd_paired(args: argparse.Namespace) -> int:
    with Path(args.detail).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    left, right = paired_accuracy(rows, args.baseline, args.treatment)
    estimate = paired_bootstrap_delta(
        left, right, iterations=args.iterations, seed=args.seed
    )
    print(
        json.dumps(
            {
                "baseline": args.baseline,
                "treatment": args.treatment,
                "n": estimate.samples,
                "bootstrap": estimate.__dict__,
                "mcnemar": mcnemar_exact(left, right),
            },
            indent=2,
        )
    )
    return 0


def cmd_clean_questions(args: argparse.Namespace) -> int:
    for path in args.files:
        report = clean_question_file(path)
        print(f"{report.path}: kept {report.kept}, removed {len(report.removed)}")
        for reason, question in report.removed:
            print(f"  [-] ({reason}) {question[:90]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser(
        "ingest", help="build a corpus from its public dataset"
    )
    ingest.add_argument("--config", required=True)
    ingest.add_argument(
        "--limit", type=int, default=1000, help="pages (PubLayNet, DocBank, DocLayNet)"
    )
    ingest.add_argument("--shards", type=int, default=None, help="PubLayNet shard cap")
    ingest.add_argument("--papers", type=int, default=100, help="SPIQA papers")
    ingest.add_argument(
        "--pool",
        type=int,
        default=300,
        help="HotpotQA questions pooled into the corpus",
    )
    ingest.add_argument(
        "--eval", type=int, default=100, help="HotpotQA questions held out"
    )
    ingest.add_argument(
        "--clean", action="store_true", help="wipe the HotpotQA corpus first"
    )
    ingest.add_argument(
        "--category", default=None, help="DocLayNet doc_category filter"
    )
    ingest.add_argument("--smoke", action="store_true", help="few items, no captioning")
    ingest.set_defaults(handler=cmd_ingest)

    questions = commands.add_parser("questions", help="author a question set")
    questions.add_argument(
        "--protocol",
        required=True,
        choices=["text", "multihop", "figure", "caption-matched", "cross-paper"],
    )
    questions.add_argument("--config", default=None)
    questions.add_argument("--corpus-dir", default=None)
    questions.add_argument("--image-dir", default=None)
    questions.add_argument("--pricing", default=None)
    questions.add_argument("--out", default=None)
    questions.add_argument("--target", type=int, default=35)
    questions.add_argument("--pixel-file", default=None)
    questions.add_argument("--graph-cache", default=None)
    questions.add_argument("--seed", type=int, default=42)
    questions.add_argument("--smoke", action="store_true")
    questions.add_argument("--verbose", action="store_true")
    questions.set_defaults(handler=cmd_questions)

    evaluate = commands.add_parser(
        "evaluate", help="run the configured systems and generators"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument(
        "--model", default=None, help="one generator instead of every configured one"
    )
    evaluate.add_argument(
        "--limit", type=int, default=None, help="first N questions (pilot)"
    )
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(handler=cmd_evaluate)

    retrieval_ab = commands.add_parser(
        "retrieval-ab", help="retrieval-only graph expansion A/B"
    )
    retrieval_ab.add_argument("--config", required=True)
    retrieval_ab.add_argument("--hops", type=int, choices=[1, 2], default=None)
    retrieval_ab.add_argument("--output", default=None)
    retrieval_ab.set_defaults(handler=cmd_retrieval_ab)

    artifacts = commands.add_parser(
        "artifacts", help="tables and figures from frozen or scanned runs"
    )
    source = artifacts.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--scan", help="results directory (exploratory only)")
    artifacts.add_argument("--out", default="artifacts")
    artifacts.add_argument("--spiqa-corpus", default="spiqa_corpus")
    artifacts.add_argument(
        "--spiqa-figures", default=str(QUESTIONS_DIR / "questions_spiqa_figures.json")
    )
    artifacts.set_defaults(handler=cmd_artifacts)

    validate = commands.add_parser(
        "validate-release", help="check a manifest and its checksums"
    )
    validate.add_argument("--manifest", required=True)
    validate.set_defaults(handler=cmd_validate)

    freeze = commands.add_parser(
        "freeze-manifest", help="write a manifest with checksums for chosen runs"
    )
    freeze.add_argument("--summary", nargs="+", required=True, help="summary CSV paths")
    freeze.add_argument("--release-id", required=True)
    freeze.add_argument("--question-set", default=None)
    freeze.add_argument("--commit", default=None)
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(handler=cmd_freeze)

    audit_graph = commands.add_parser(
        "audit-graph", help="sample triples for human audit"
    )
    audit_graph.add_argument("cache")
    audit_graph.add_argument("--corpus", required=True)
    audit_graph.add_argument("--count", type=int, default=200)
    audit_graph.add_argument("--seed", type=int, default=20260903)
    audit_graph.add_argument(
        "--output", default="protocols/graph_triple_audit_sample.csv"
    )
    audit_graph.set_defaults(handler=cmd_audit_graph)

    audit_figures = commands.add_parser(
        "audit-figures", help="screen visual questions for text leakage"
    )
    audit_figures.add_argument("questions")
    audit_figures.add_argument("captions")
    audit_figures.add_argument("corpus")
    audit_figures.add_argument(
        "--output", default="artifacts/runs/figure_integrity.csv"
    )
    audit_figures.set_defaults(handler=cmd_audit_figures)

    paired = commands.add_parser(
        "paired-analysis", help="paired bootstrap and McNemar for one contrast"
    )
    paired.add_argument("detail")
    paired.add_argument("--baseline", default="baseline")
    paired.add_argument("--treatment", required=True)
    paired.add_argument("--iterations", type=int, default=10000)
    paired.add_argument("--seed", type=int, default=20260903)
    paired.set_defaults(handler=cmd_paired)

    clean = commands.add_parser(
        "clean-questions", help="drop items that fail the current validators"
    )
    clean.add_argument("files", nargs="+")
    clean.set_defaults(handler=cmd_clean_questions)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except MultimodalGraphRagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
