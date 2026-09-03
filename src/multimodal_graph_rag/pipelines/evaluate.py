"""Run every configured system on one question set with one generator.

Retrieval components are built once and shared by all systems. Each question
is scored for retrieval (partial recall, full-provenance completeness, MRR),
answer accuracy, faithfulness, and relevancy, and every generation call's
tokens, latency, and cost are recorded. A provider failure stops the run: the
rows completed so far are written with a ``partial`` prefix and the error is
raised, so a rate limit can never be scored as a zero.

Outputs (never overwritten)
    <results_dir>/summary_<qset>_<model>_<stamp>.csv   one row per system
    <results_dir>/detail_<qset>_<model>_<stamp>.csv    one row per question
    <results_dir>/trace_<qset>_<model>_<stamp>.jsonl   one RunRecord per
                                                       question and system,
                                                       with the evidence bundle
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..artifacts.manifest import file_sha256
from ..clients import DIAGNOSTICS, GENERATORS, ModelClient
from ..config import ExperimentConfig
from ..errors import ConfigurationError, MultimodalGraphRagError
from ..evaluation.judges import Judges
from ..retrieval.graph import default_cache_path, load_graph
from ..retrieval.metrics import retrieval_metrics
from ..retrieval.text import TextIndex
from ..retrieval.visual import ClipImageIndex
from ..schemas import QuestionRecord, RunRecord, content_hash, load_questions
from . import EvidenceControl
from .systems import (
    GRAPH_SYSTEMS,
    IMAGE_SYSTEMS,
    SYSTEMS,
    RunContext,
    SystemOutput,
    resolve_systems,
)

QUALITY_METRICS = ("recall", "complete", "mrr", "acc", "faith", "rel")
EFFICIENCY_METRICS = ("in_tok", "out_tok", "latency_ms", "cost_usd")
SUMMARY_FIELDS = (
    "run_id",
    "experiment",
    "question_set",
    "question_type",
    "model",
    "vendor",
    "access",
    "n_questions",
    "system",
    "config_hash",
    "question_file_sha256",
    "corpus_dir",
    "image_dir",
    "candidate_budget",
    "bridge_base_k",
    "bridge_slots",
    "bridge_hops",
    "vision_mode",
    "pricing_captured_on",
    "recall",
    "complete",
    "mrr",
    "acc",
    "faith",
    "rel",
    "mean_in_tok",
    "mean_out_tok",
    "mean_latency_ms",
    "total_cost_usd",
    "cost_per_100q_usd",
    "judge_cost_usd",
    "n_errors",
    "error_rate",
)


@dataclass(frozen=True)
class EvaluationSettings:
    question_file: Path
    model: str
    systems: tuple[str, ...]
    corpus_dir: Path
    pricing_captured_on: str
    image_dir: Path | None = None
    candidate_budget: int = 5
    bridge_hops: int = 1
    vision_mode: str = "pixels"
    text_judge: str = "deepseek"
    vision_judge: str = "claude-haiku"
    cache_dir: Path = Path(".cache")
    graph_cache_dir: Path = Path("data/graphs")
    results_dir: Path = Path("artifacts/runs")
    seed: int = 0
    limit: int | None = None
    experiment: str = "ad-hoc"
    experiment_hash: str = ""
    extra_identity: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model not in GENERATORS + DIAGNOSTICS:
            raise ConfigurationError(
                f"{self.model} is not a generator; choose from {GENERATORS + DIAGNOSTICS}"
            )
        if self.candidate_budget < 3:
            raise ConfigurationError("candidate_budget must be at least 3")
        if self.vision_mode not in ("pixels", "captions"):
            raise ConfigurationError("vision_mode must be 'pixels' or 'captions'")
        if self.limit is not None and self.limit < 1:
            raise ConfigurationError("limit must be positive")
        resolve_systems(self.systems)

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        model: str,
        *,
        pricing_captured_on: str,
        limit: int | None = None,
        root: Path = Path("."),
    ) -> EvaluationSettings:
        controls = [
            EvidenceControl(control).system_name
            for control in config.controls
            if control != EvidenceControl.NORMAL.value
        ]
        return cls(
            question_file=root / config.question_set,
            model=model,
            systems=tuple(list(config.systems) + controls),
            corpus_dir=root / config.corpus_dir,
            pricing_captured_on=pricing_captured_on,
            image_dir=(root / config.image_dir) if config.image_dir else None,
            candidate_budget=config.candidate_budget,
            bridge_hops=config.bridge_hops,
            vision_mode=config.vision_mode,
            text_judge=config.judge,
            vision_judge=config.vision_judge,
            cache_dir=root / config.cache_dir,
            results_dir=root / config.results_dir,
            seed=config.seed,
            limit=limit,
            experiment=config.name,
            experiment_hash=config.config_hash,
            extra_identity={
                "graph_version": config.graph_version,
                "prompt_version": config.prompt_version,
                "visual_retriever": config.visual_retriever,
                "repetition": config.repetition,
            },
        )

    @property
    def bridge_base_k(self) -> int:
        return min(3, self.candidate_budget)

    @property
    def question_set_name(self) -> str:
        name = self.question_file.stem
        return name if self.vision_mode == "pixels" else f"{name}_captions"


@dataclass(frozen=True)
class RunOutputs:
    run_id: str
    config_hash: str
    summary_path: Path
    detail_path: Path
    trace_path: Path


@dataclass
class _Aggregate:
    totals: dict[str, float] = field(
        default_factory=lambda: {m: 0.0 for m in QUALITY_METRICS + EFFICIENCY_METRICS}
    )
    judge_cost: float = 0.0


def _score_retrieval(
    output: SystemOutput, question: QuestionRecord
) -> tuple[float, int, float]:
    ranked = (
        output.ranked_image_sources
        if question.is_visual
        else output.ranked_text_sources
    )
    if not ranked:
        return 0.0, 0, 0.0
    metrics = retrieval_metrics(ranked, question.gold_sources, len(ranked))
    return (
        metrics[f"recall@{len(ranked)}"],
        int(metrics["completeness"]),
        metrics["mrr"],
    )


def _guard_corpus(
    questions: Sequence[QuestionRecord], index: TextIndex, corpus_dir: Path
) -> None:
    text_golds = {source for q in questions for source in q.text_gold_sources}
    if text_golds and not (text_golds & index.source_names):
        raise ConfigurationError(
            f"none of the questions' gold documents exist in {corpus_dir}; "
            "the question set and corpus do not match"
        )


def _write_rows(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(settings: EvaluationSettings, client: ModelClient) -> RunOutputs:
    systems = resolve_systems(settings.systems)
    questions = load_questions(settings.question_file)
    if settings.limit:
        questions = questions[: settings.limit]
    spec = client.spec(settings.model)
    judges = Judges(
        client, text_judge=settings.text_judge, vision_judge=settings.vision_judge
    )

    now = datetime.now(UTC)
    run_id = now.isoformat(timespec="seconds")
    stamp = now.strftime("%Y%m%d_%H%M%S")
    question_sha = file_sha256(settings.question_file)
    identity = {
        "experiment": settings.experiment,
        "experiment_hash": settings.experiment_hash,
        "question_set": settings.question_set_name,
        "question_file_sha256": question_sha,
        "model": settings.model,
        "systems": systems,
        "corpus_dir": str(settings.corpus_dir.resolve()),
        "image_dir": str(settings.image_dir.resolve()) if settings.image_dir else "",
        "candidate_budget": settings.candidate_budget,
        "bridge_base_k": settings.bridge_base_k,
        "bridge_slots": settings.candidate_budget - settings.bridge_base_k,
        "bridge_hops": settings.bridge_hops,
        "vision_mode": settings.vision_mode,
        "text_judge": settings.text_judge,
        "vision_judge": settings.vision_judge,
        "seed": settings.seed,
        "limit": settings.limit,
        **settings.extra_identity,
    }
    config_hash = content_hash(identity)

    print(f"\nEvaluation - {settings.question_set_name} ({len(questions)} questions)")
    print(f"Generator : {settings.model} ({spec.vendor}, {spec.access})")
    if spec.access == "diagnostic":
        print("            DIAGNOSTIC RUN - excluded from the reported comparison")
    print(f"Systems   : {', '.join(systems)}")
    print(f"Judges    : {settings.text_judge} (text), {settings.vision_judge} (vision)")
    print(f"Corpus    : {settings.corpus_dir}")
    print(f"Identity  : {config_hash[:16]}")

    text_index = TextIndex.build(
        settings.corpus_dir, client, cache_dir=settings.cache_dir
    )
    _guard_corpus(questions, text_index, settings.corpus_dir)
    graph = None
    if GRAPH_SYSTEMS & set(systems):
        graph = load_graph(
            settings.corpus_dir,
            default_cache_path(settings.corpus_dir, settings.graph_cache_dir),
            client,
        )
        stats = graph.statistics()
        print(f"Graph     : {stats.node_count} nodes, {stats.edge_count} edges")
    image_index = None
    if IMAGE_SYSTEMS & set(systems):
        if settings.image_dir is None:
            raise ConfigurationError("image systems require image_dir")
        image_index = ClipImageIndex.build(
            settings.image_dir, cache_dir=settings.cache_dir
        )
        print(f"Images    : {image_index.size} crops, mode={settings.vision_mode}")

    context = RunContext(
        client=client,
        model=settings.model,
        text_index=text_index,
        candidate_budget=settings.candidate_budget,
        graph=graph,
        image_index=image_index,
        image_dir=settings.image_dir,
        bridge_base_k=settings.bridge_base_k,
        bridge_hops=settings.bridge_hops,
        seed=settings.seed,
    )

    results_dir = settings.results_dir
    base_name = f"{settings.question_set_name}_{settings.model}_{stamp}"
    summary_path = results_dir / f"summary_{base_name}.csv"
    detail_path = results_dir / f"detail_{base_name}.csv"
    trace_path = results_dir / f"trace_{base_name}.jsonl"

    aggregates = {name: _Aggregate() for name in systems}
    detail_rows: list[dict[str, object]] = []
    try:
        for number, question in enumerate(questions, 1):
            use_pixels = settings.vision_mode == "pixels" and question.is_visual
            row: dict[str, object] = {
                "question_id": question.id,
                "question": question.question,
                "expected_source": "|".join(question.gold_sources),
                "question_type": question.question_type,
                "model": settings.model,
            }
            for name in systems:
                output = SYSTEMS[name](question, context, use_pixels)
                recall, complete, mrr = _score_retrieval(output, question)
                answer = judges.answer(
                    question.question, question.answer, output.result.text
                )
                faith = judges.faithfulness(
                    output.result.text,
                    output.context,
                    image_paths=output.image_paths if question.is_visual else (),
                )
                relevancy = judges.relevancy(question.question, output.result.text)
                values = {
                    "recall": recall,
                    "complete": complete,
                    "mrr": mrr,
                    "acc": answer.value,
                    "faith": faith.value,
                    "rel": relevancy.value,
                    "in_tok": output.result.input_tokens,
                    "out_tok": output.result.output_tokens,
                    "latency_ms": output.result.latency_ms,
                    "cost_usd": output.result.cost_usd,
                }
                aggregate = aggregates[name]
                for metric, value in values.items():
                    aggregate.totals[metric] += float(value)
                aggregate.judge_cost += (
                    answer.call.cost_usd + faith.call.cost_usd + relevancy.call.cost_usd
                )
                row.update(
                    {
                        f"{name}_acc": answer.value,
                        f"{name}_faith": faith.value,
                        f"{name}_rel": relevancy.value,
                        f"{name}_rec": round(recall, 3),
                        f"{name}_complete": complete,
                        f"{name}_mrr": round(mrr, 3),
                        f"{name}_in_tok": output.result.input_tokens,
                        f"{name}_out_tok": output.result.output_tokens,
                        f"{name}_latency_ms": round(output.result.latency_ms, 1),
                        f"{name}_cost_usd": round(output.result.cost_usd, 6),
                        f"{name}_answer": output.result.text.replace("\n", " ")[:160],
                        f"{name}_sources": "|".join(output.ranked_text_sources),
                    }
                )
                RunRecord(
                    run_id=run_id,
                    config_hash=config_hash,
                    question_id=question.id,
                    system=name,
                    answer=output.result.text,
                    evidence=output.evidence_bundle(settings.candidate_budget),
                    metrics={k: v for k, v in values.items() if k in QUALITY_METRICS},
                    raw_judges={
                        "accuracy": answer.to_dict(),
                        "faithfulness": faith.to_dict(),
                        "relevancy": relevancy.to_dict(),
                    },
                    input_tokens=output.result.input_tokens,
                    output_tokens=output.result.output_tokens,
                    latency_ms=output.result.latency_ms,
                    cost_usd=output.result.cost_usd,
                    model_versions={settings.model: spec.model},
                ).write_jsonl(trace_path)
            detail_rows.append(row)
            print(f"  {number:3d}/{len(questions)}  {question.question[:58]}")
    except MultimodalGraphRagError:
        if detail_rows:
            partial = results_dir / f"detail_partial_{base_name}.csv"
            _write_rows(partial, list(detail_rows[0]), detail_rows)
            print(
                f"\nrun aborted after {len(detail_rows)} questions; partial rows -> {partial}"
            )
        raise

    count = len(questions)
    summary_rows = []
    for name in systems:
        totals = aggregates[name].totals
        summary_rows.append(
            {
                "run_id": run_id,
                "experiment": settings.experiment,
                "question_set": settings.question_set_name,
                "question_type": questions[0].question_type,
                "model": settings.model,
                "vendor": spec.vendor,
                "access": spec.access,
                "n_questions": count,
                "system": name,
                "config_hash": config_hash,
                "question_file_sha256": question_sha,
                "corpus_dir": str(settings.corpus_dir),
                "image_dir": str(settings.image_dir) if image_index else "",
                "candidate_budget": settings.candidate_budget,
                "bridge_base_k": settings.bridge_base_k,
                "bridge_slots": settings.candidate_budget - settings.bridge_base_k,
                "bridge_hops": settings.bridge_hops,
                "vision_mode": settings.vision_mode,
                "pricing_captured_on": settings.pricing_captured_on,
                "recall": round(totals["recall"] / count, 3),
                "complete": round(totals["complete"] / count, 3),
                "mrr": round(totals["mrr"] / count, 3),
                "acc": round(totals["acc"] / count, 3),
                "faith": round(totals["faith"] / count, 3),
                "rel": round(totals["rel"] / count, 3),
                "mean_in_tok": round(totals["in_tok"] / count, 1),
                "mean_out_tok": round(totals["out_tok"] / count, 1),
                "mean_latency_ms": round(totals["latency_ms"] / count, 1),
                "total_cost_usd": round(totals["cost_usd"], 6),
                "cost_per_100q_usd": round(totals["cost_usd"] / count * 100, 4),
                "judge_cost_usd": round(aggregates[name].judge_cost, 6),
                "n_errors": 0,
                "error_rate": 0.0,
            }
        )
    _write_rows(summary_path, SUMMARY_FIELDS, summary_rows)
    _write_rows(detail_path, list(detail_rows[0]), detail_rows)

    print(f"\n{'metric':<14}" + "".join(f"{name:>16}" for name in systems))
    for metric in QUALITY_METRICS:
        print(
            f"{metric:<14}" + "".join(f"{row[metric]:>16.3f}" for row in summary_rows)
        )
    print(
        f"{'cost/100q $':<14}"
        + "".join(f"{row['cost_per_100q_usd']:>16.4f}" for row in summary_rows)
    )
    print(
        f"\nSummary -> {summary_path}\nDetail  -> {detail_path}\nTrace   -> {trace_path}"
    )
    return RunOutputs(run_id, config_hash, summary_path, detail_path, trace_path)
