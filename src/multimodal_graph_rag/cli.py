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
from dataclasses import replace
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
from .clients import GENERATOR_MAX_TOKENS, GENERATORS, ModelClient, load_pricing
from .config import ExperimentConfig, load_experiment_config
from .corpora import corpus_spec, docbank, doclaynet, hotpotqa, publaynet, spiqa
from .errors import ConfigurationError, MultimodalGraphRagError
from .evaluation.answer_match import normalise_answer
from .evaluation.audits import (
    FIGURE_AUDIT_FIELDS,
    TRIPLE_AUDIT_FIELDS,
    figure_integrity_rows,
    paired_accuracy,
    sample_triple_audit,
)
from .evaluation.calibration import calibration_items, run_calibration
from .evaluation.graph_grounding import (
    audit_triple_grounding,
    canonicalisation_collisions,
)
from .evaluation.multimodal import (
    CAPTIONS,
    PIXELS,
    analyse,
    gold_sent,
    merge_caption_arms,
    text_recoverable,
)
from .evaluation.pooled import adjust_family, compare, load_outcomes
from .evaluation.posthoc import (
    PASSES,
    JudgementStore,
    assign_states,
    judge_context,
    judgements_path,
    load_trace,
    plan_tasks,
    run_task,
)
from .evaluation.statistics import mcnemar_exact, paired_bootstrap_delta
from .pipelines.evaluate import (
    LEGACY_GENERATOR_MAX_TOKENS,
    EvaluationSettings,
    capped_records,
    run_evaluation,
)
from .pipelines.systems import (
    COMPARATOR_SYSTEMS,
    PASSAGE_RETRIEVERS,
    RunContext,
    passage_arm_report,
)
from .questions import clean_question_file
from .questions.authoring import author_figure_questions, author_text_questions
from .questions.caption_matched import author_caption_matched
from .questions.cross_paper import author_cross_paper
from .questions.spiqa_native import convert_spiqa_native
from .retrieval.baselines import chunk_lexicon
from .retrieval.comparators import PrecomputedRetrievals, comparator_report
from .retrieval.graph import TripleStore, default_cache_path, load_graph
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
            # The released corpus and spiqa_gold_qa.json were sampled at the
            # default seed; the configs' seed samples questions, not papers.
            seed=spiqa.DEFAULT_SEED,
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


def _print_question_report(report: object) -> None:
    print(
        json.dumps(
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in report.__dict__.items()
            },
            indent=2,
        )
    )


def cmd_questions(args: argparse.Namespace) -> int:
    config: ExperimentConfig | None = (
        load_experiment_config(args.config) if args.config else None
    )
    corpus_dir = args.corpus_dir or (config.corpus_dir if config else None)
    image_dir = args.image_dir or (config.image_dir if config else None)

    # SPIQA's curated QA is already written; converting it calls no provider,
    # so it needs neither a pricing snapshot nor a client.
    if args.protocol == "spiqa-native":
        if not args.gold_file or not image_dir or not args.out:
            raise ConfigurationError(
                "spiqa-native conversion needs --gold-file, --image-dir, and --out"
            )
        report = convert_spiqa_native(
            args.gold_file,
            Path(image_dir) / "captions.json",
            args.out,
            author_captions_file=Path(image_dir) / "author_captions.json",
            target=args.target if args.target > 0 else None,
            seed=args.seed,
            exclude_flagged=args.exclude_flagged,
            only_flagged=args.only_flagged,
        )
        _print_question_report(report)
        return 0

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
    _print_question_report(report)
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
    if args.regenerate_capped:
        return _regenerate_capped(args, config)
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
    if args.resume and not args.model:
        raise ConfigurationError("--resume continues one generator's run; pass --model")
    pricing = load_pricing(config.pricing_snapshot)
    client = ModelClient(pricing)
    for model in models:
        settings = EvaluationSettings.from_config(
            config, model, pricing_captured_on=pricing.captured_on, limit=args.limit
        )
        if args.budget_usd is not None:
            settings = replace(settings, budget_usd=args.budget_usd)
        run_evaluation(
            settings, client, resume_trace=Path(args.resume) if args.resume else None
        )
    return 0


def _regenerate_capped(args: argparse.Namespace, config: ExperimentConfig) -> int:
    if not (args.resume and args.model):
        raise ConfigurationError(
            "--regenerate-capped amends one finished run; pass --resume and --model"
        )
    trace = Path(args.resume)
    pricing = load_pricing(config.pricing_snapshot)
    capped = capped_records(load_trace(trace))
    price = pricing.price(args.model)
    systems: dict[str, int] = {}
    for record in capped:
        systems[record["system"]] = systems.get(record["system"], 0) + 1
    # A capped reply already used the old cap, so a new one costs at least that
    # much and at most the current cap; the two judge calls add little.
    floor = sum(
        price.cost(int(r["input_tokens"]), LEGACY_GENERATOR_MAX_TOKENS) for r in capped
    )
    ceiling = sum(
        price.cost(int(r["input_tokens"]), GENERATOR_MAX_TOKENS) for r in capped
    )
    print(
        f"{trace.name}: {len(capped)} answers reached the "
        f"{LEGACY_GENERATOR_MAX_TOKENS}-token cap {systems}"
    )
    print(
        f"  generation ${floor:.2f} to ${ceiling:.2f} with {args.model}, "
        f"plus {2 * len(capped)} judge calls"
    )
    if args.dry_run:
        return 0
    settings = EvaluationSettings.from_config(
        config, args.model, pricing_captured_on=pricing.captured_on, limit=args.limit
    )
    run_evaluation(
        settings, ModelClient(pricing), resume_trace=trace, regenerate_capped=True
    )
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
    base_k = min(3, config.candidate_budget)
    report = bridge_report(
        questions,
        index,
        graph,
        client,
        k=base_k,
        n_bridge=config.candidate_budget - base_k,
        hops=hops,
    )
    comparator = None
    arms = [arm for arm in PASSAGE_RETRIEVERS if arm not in COMPARATOR_SYSTEMS]
    if config.comparator_retrievals and Path(config.comparator_retrievals).is_file():
        comparator = PrecomputedRetrievals.load(
            config.comparator_retrievals, corpus_dir=config.corpus_dir
        )
        arms.extend(sorted(COMPARATOR_SYSTEMS))
    elif config.comparator_retrievals:
        print(f"skipping +HippoRAG: {config.comparator_retrievals} does not exist yet")
    # Retrieval only: nothing is generated, so no generator is named.
    context = RunContext(
        client=client,
        model="",
        text_index=index,
        candidate_budget=config.candidate_budget,
        graph=graph,
        comparator=comparator,
        lexicon=chunk_lexicon(index),
        bridge_base_k=base_k,
        bridge_hops=hops,
        seed=config.seed,
    )
    arm_report = passage_arm_report(questions, context, arms)
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
                "arms": arm_report,
            },
            stream,
            indent=2,
        )
    print(json.dumps(report.summary(), indent=2))
    print(json.dumps(arm_report["summary"], indent=2))
    print(f"wrote {output}")
    return 0


def cmd_comparator_ab(args: argparse.Namespace) -> int:
    """Retrieval-only completeness of precomputed comparator rankings."""
    questions = load_questions(args.questions)
    retrievals = PrecomputedRetrievals.load(args.retrievals, corpus_dir=args.corpus_dir)
    report = comparator_report(questions, retrievals, k=args.k)
    output = Path(
        args.output
        or f"artifacts/runs/comparator_ab_{Path(args.questions).stem}_k{args.k}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report["summary"], indent=2))
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


def cmd_graph_grounding(args: argparse.Namespace) -> int:
    report = audit_triple_grounding(args.corpus_dir, args.cache, corpus=args.corpus)
    payload = report.to_dict()
    if args.collisions:
        store = TripleStore.load(args.cache)
        collisions = canonicalisation_collisions(store.entries)
        payload["surface_forms_merged_into_one_node"] = len(collisions)
        payload["example_collisions"] = dict(list(collisions.items())[:10])
    print(json.dumps(payload, indent=2))
    print(
        "\nGrounding is not precision: it shows the endpoints occur in the "
        "source chunk, not that the relation is correct."
    )
    return 0


def cmd_judge_pass(args: argparse.Namespace) -> int:
    questions = {q.id: q for q in load_questions(args.questions)}
    pricing = load_pricing(args.pricing)
    price = pricing.price(args.judge)
    client = None if args.dry_run else ModelClient(pricing)
    total_calls, total_estimate = 0, 0.0
    for trace in args.traces:
        records = [r for r in load_trace(trace) if not r.get("error")]
        missing = {r["question_id"] for r in records} - set(questions)
        if missing:
            raise ConfigurationError(
                f"{trace} has questions not in {args.questions}: {sorted(missing)[:3]}"
            )
        store = JudgementStore(judgements_path(trace))
        tasks = plan_tasks(
            args.pass_name,
            records,
            questions,
            store,
            fraction=args.fraction,
            seed=args.seed,
            repeats=args.repeats,
            correctness=args.correctness,
        )
        # Rough price: about four characters per token, a 1,500-token allowance
        # per attached image, and 150 output tokens for a reasoned verdict.
        estimate = sum(
            price.cost(
                (len(judge_context(t.record)) + len(t.record["answer"]) + 600) // 4
                + 1500
                * (args.vision_mode == "pixels")
                * (args.pass_name in ("second-faithfulness", "sufficiency")),
                150,
            )
            for t in tasks
        )
        total_calls += len(tasks)
        total_estimate += estimate
        print(f"{Path(trace).name}: {len(tasks)} calls, about ${estimate:.2f}")
        if args.dry_run:
            continue
        for number, task in enumerate(tasks, 1):
            store.add(
                run_task(
                    task,
                    questions[task.record["question_id"]],
                    client,
                    args.judge,
                    args.image_dir,
                    args.vision_mode,
                )
            )
            if number % 50 == 0:
                print(f"  {number}/{len(tasks)}", flush=True)
    print(f"total: {total_calls} calls, about ${total_estimate:.2f} with {args.judge}")
    return 0


def cmd_pooled_analysis(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions)
    scorable = [q for q in questions if normalise_answer(q.answer)]
    if len(scorable) < len(questions):
        skipped = [q.id for q in questions if not normalise_answer(q.answer)]
        print(
            f"excluded {len(skipped)} question(s) with no reference answer: {skipped}"
        )
    splits = []
    for pair in args.split:
        treatment, _, reference = pair.partition("=")
        if not reference:
            raise ConfigurationError(f"--split takes TREATMENT=REFERENCE, not {pair!r}")
        splits.append((treatment, reference))
    report = compare(
        load_outcomes(args.traces, scorable),
        args.reference,
        splits=splits,
        iterations=args.iterations,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        f"{report['n_questions']} questions x {len(report['generators'])} generators, "
        f"reference {args.reference}"
    )
    for system, entry in report["systems"].items():
        line = (
            f"  {system:<22} complete {entry['complete']:.2f}  judge {entry['acc']:.3f}"
        )
        difference = entry["differences"].get("acc")
        if difference:
            line += (
                f"  delta {difference['delta']:+.3f} "
                f"[{difference['lower']:+.3f}, {difference['upper']:+.3f}] "
                f"p {difference['p']:.4f}"
            )
        print(line)
    print(f"report -> {output}")
    return 0


def cmd_multimodal_analysis(args: argparse.Namespace) -> int:
    screened_questions = load_questions(args.screened_questions)
    flagged_questions = load_questions(args.flagged_questions)
    obscure_questions = load_questions(args.publaynet_questions)
    screened = merge_caption_arms(
        load_outcomes(args.screened, screened_questions),
        load_outcomes(args.captions, screened_questions),
    )
    report = analyse(
        flagged=load_outcomes(args.flagged, flagged_questions),
        screened=screened,
        obscure=load_outcomes(args.publaynet, obscure_questions),
        recoverable_from_text=text_recoverable(args.integrity),
        flagged_sent=gold_sent(args.flagged, flagged_questions, PIXELS),
        screened_sent=gold_sent(args.screened, screened_questions, PIXELS),
        obscure_sent=gold_sent(args.publaynet, obscure_questions, PIXELS),
        iterations=args.iterations,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, column in report["columns"].items():
        arms = "  ".join(
            f"{system} {entry['acc']:.3f}" for system, entry in column["arms"].items()
        )
        print(f"{name} (n={column['n']}): {arms}")
    for name, test in report["family"].items():
        print(
            f"  {name:<32} {test['delta']:+.3f} [{test['lower']:+.3f}, "
            f"{test['upper']:+.3f}] p {test['p']:.4f} Holm {test['p_holm']:.4f}"
        )
    print(f"caption arm in the report as {CAPTIONS}; report -> {output}")
    return 0


def cmd_judge_calibration(args: argparse.Namespace) -> int:
    items = calibration_items(load_questions(args.questions))
    pricing = load_pricing(args.pricing)
    price = pricing.price(args.judge)
    # about four characters per token, and 150 output tokens per verdict
    estimate = sum(price.cost(len(item.prompt) // 4, 150) for item in items)
    print(f"{len(items)} calibration calls, about ${estimate:.2f} with {args.judge}")
    if args.dry_run:
        return 0
    summary = run_calibration(items, ModelClient(pricing), args.judge, args.output)
    print(json.dumps(summary, indent=1))
    return 0


def cmd_holm_family(args: argparse.Namespace) -> int:
    paths = [Path(path) for path in args.reports]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    adjust_family(reports)
    for path, report in zip(paths, reports, strict=True):
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Holm-adjusted {len(paths)} reports as one family")
    return 0


def cmd_evidence_states(args: argparse.Namespace) -> int:
    questions = {q.id: q for q in load_questions(args.questions)}
    available = {p.name for p in Path(args.corpus_dir).iterdir()}
    if args.image_dir:
        available |= {p.name for p in Path(args.image_dir).iterdir()}
    rows = []
    for trace in args.traces:
        records = [r for r in load_trace(trace) if not r.get("error")]
        store = JudgementStore(judgements_path(trace))
        for row in assign_states(
            records,
            questions,
            store,
            available_sources=available,
            correctness=args.correctness,
        ):
            rows.append({"trace": Path(trace).name, **row})
    fields = [
        "trace",
        "question_id",
        "system",
        "correct",
        "gold_received",
        "gold_required",
        "state",
    ]
    _write_csv(Path(args.output), fields, rows)
    pending = sum(row["state"] == "pending_sufficiency" for row in rows)
    print(f"wrote {args.output}: {len(rows)} answers, {pending} awaiting sufficiency")
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
        choices=[
            "text",
            "multihop",
            "figure",
            "caption-matched",
            "cross-paper",
            "spiqa-native",
        ],
    )
    questions.add_argument("--config", default=None)
    questions.add_argument("--corpus-dir", default=None)
    questions.add_argument("--image-dir", default=None)
    questions.add_argument("--pricing", default=None)
    questions.add_argument("--out", default=None)
    questions.add_argument(
        "--target",
        type=int,
        default=35,
        help="how many questions to write; 0 means every eligible item "
        "(spiqa-native only)",
    )
    questions.add_argument("--pixel-file", default=None)
    questions.add_argument("--graph-cache", default=None)
    questions.add_argument(
        "--gold-file",
        default=None,
        help="spiqa-native: the curated QA written by `rag ingest`",
    )
    questions.add_argument(
        "--exclude-flagged",
        action="store_true",
        help="spiqa-native: sample only items the leakage screen did not flag",
    )
    questions.add_argument(
        "--only-flagged",
        action="store_true",
        help="spiqa-native: keep only the items the leakage screen flagged",
    )
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
    evaluate.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="stop this generator's run past this spend; not part of its identity",
    )
    evaluate.add_argument(
        "--resume",
        default=None,
        help="trace_*.jsonl of a stopped run to carry on (same config and --model)",
    )
    evaluate.add_argument(
        "--regenerate-capped",
        action="store_true",
        help=(
            "answer again, at the current generator cap, every answer in the "
            f"--resume trace that reached the old {LEGACY_GENERATOR_MAX_TOKENS}-token "
            "cap (--dry-run prices it)"
        ),
    )
    evaluate.set_defaults(handler=cmd_evaluate)

    retrieval_ab = commands.add_parser(
        "retrieval-ab", help="retrieval-only comparison of every passage arm"
    )
    retrieval_ab.add_argument("--config", required=True)
    retrieval_ab.add_argument("--hops", type=int, choices=[1, 2], default=None)
    retrieval_ab.add_argument("--output", default=None)
    retrieval_ab.set_defaults(handler=cmd_retrieval_ab)

    comparator_ab = commands.add_parser(
        "comparator-ab",
        help="retrieval-only completeness of precomputed comparator rankings",
    )
    comparator_ab.add_argument("--questions", required=True)
    comparator_ab.add_argument("--retrievals", required=True)
    comparator_ab.add_argument("--corpus-dir", default=None)
    comparator_ab.add_argument("--k", type=int, default=5)
    comparator_ab.add_argument("--output", default=None)
    comparator_ab.set_defaults(handler=cmd_comparator_ab)

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
        "audit-graph", help="sample triples for an optional human audit"
    )
    audit_graph.add_argument("cache")
    audit_graph.add_argument("--corpus", required=True)
    audit_graph.add_argument("--count", type=int, default=200)
    audit_graph.add_argument("--seed", type=int, default=20260903)
    audit_graph.add_argument(
        "--output", default="protocols/graph_triple_audit_sample.csv"
    )
    audit_graph.set_defaults(handler=cmd_audit_graph)

    grounding = commands.add_parser(
        "graph-grounding",
        help="check every extracted triple against its source chunk",
    )
    grounding.add_argument("cache")
    grounding.add_argument("--corpus-dir", required=True)
    grounding.add_argument("--corpus", default=None)
    grounding.add_argument(
        "--collisions",
        action="store_true",
        help="also count surface forms that merge into one graph node",
    )
    grounding.set_defaults(handler=cmd_graph_grounding)

    judge_pass = commands.add_parser(
        "judge-pass",
        help="run a judging pass over saved run records (resumable; --dry-run prices it)",
    )
    judge_pass.add_argument("traces", nargs="+", help="trace_*.jsonl files")
    judge_pass.add_argument("--questions", required=True)
    judge_pass.add_argument("--pass", dest="pass_name", required=True, choices=PASSES)
    judge_pass.add_argument("--judge", required=True)
    judge_pass.add_argument("--fraction", type=float, default=1.0)
    judge_pass.add_argument("--repeats", type=int, default=3)
    judge_pass.add_argument("--seed", type=int, default=20260903)
    judge_pass.add_argument("--correctness", choices=("judge", "em"), default="judge")
    judge_pass.add_argument("--image-dir", default=None)
    judge_pass.add_argument(
        "--vision-mode", choices=("pixels", "captions"), default="captions"
    )
    judge_pass.add_argument("--pricing", default="configs/pricing_2026-10-20.json")
    judge_pass.add_argument("--dry-run", action="store_true")
    judge_pass.set_defaults(handler=cmd_judge_pass)

    pooled = commands.add_parser(
        "pooled-analysis",
        help="compare every system with a reference, pooled over generators",
    )
    pooled.add_argument("traces", nargs="+", help="one trace_*.jsonl per generator")
    pooled.add_argument("--questions", required=True)
    pooled.add_argument("--reference", default="baseline")
    pooled.add_argument(
        "--split",
        action="append",
        default=[],
        help="TREATMENT=REFERENCE pair to split into access and use terms",
    )
    pooled.add_argument("--iterations", type=int, default=10_000)
    pooled.add_argument("--seed", type=int, default=20261022)
    pooled.add_argument("--output", required=True)
    pooled.set_defaults(handler=cmd_pooled_analysis)

    multimodal = commands.add_parser(
        "multimodal-analysis",
        help="the RQ3 table and the tests of H3, from the figure-set traces",
    )
    multimodal.add_argument("--screened", nargs="+", required=True)
    multimodal.add_argument("--captions", nargs="+", required=True)
    multimodal.add_argument("--flagged", nargs="+", required=True)
    multimodal.add_argument("--publaynet", nargs="+", required=True)
    multimodal.add_argument("--screened-questions", required=True)
    multimodal.add_argument("--flagged-questions", required=True)
    multimodal.add_argument("--publaynet-questions", required=True)
    multimodal.add_argument("--integrity", required=True)
    multimodal.add_argument("--iterations", type=int, default=10_000)
    multimodal.add_argument("--seed", type=int, default=20261022)
    multimodal.add_argument("--output", required=True)
    multimodal.set_defaults(handler=cmd_multimodal_analysis)

    calibration = commands.add_parser(
        "judge-calibration",
        help="grade each reference and a decoy reference to bound judge error",
    )
    calibration.add_argument("--questions", required=True)
    calibration.add_argument("--judge", required=True)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--pricing", default="configs/pricing_2026-10-20.json")
    calibration.add_argument("--dry-run", action="store_true")
    calibration.set_defaults(handler=cmd_judge_calibration)

    family = commands.add_parser(
        "holm-family",
        help="re-adjust several pooled-analysis reports as one Holm family",
    )
    family.add_argument("reports", nargs="+")
    family.set_defaults(handler=cmd_holm_family)

    states = commands.add_parser(
        "evidence-states", help="assign an evidence state to every saved answer"
    )
    states.add_argument("traces", nargs="+", help="trace_*.jsonl files")
    states.add_argument("--questions", required=True)
    states.add_argument("--corpus-dir", required=True)
    states.add_argument("--image-dir", default=None)
    states.add_argument("--correctness", choices=("judge", "em"), default="judge")
    states.add_argument("--output", required=True)
    states.set_defaults(handler=cmd_evidence_states)

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
