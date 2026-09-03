"""HotpotQA ingestion: an entity-linked corpus with native bridge and comparison sets.

Paragraphs from ``pool`` questions (two gold plus eight distractors each) form
the corpus; ``eval`` questions are held out, half bridge and half comparison.
Pooling from more questions than are evaluated makes dense retrieval miss a
gold paragraph often enough for graph expansion to have a bridge to build.

Produces
    <corpus_dir>/<title>.txt                       one paragraph per file
    <questions_dir>/questions_hotpotqa_bridge.json
    <questions_dir>/questions_hotpotqa_comparison.json
"""

from __future__ import annotations

import hashlib
import random
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from ..schemas import QuestionRecord, write_questions

HF_DATASET = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
DEFAULT_SEED = 42
QUESTION_TYPES = ("bridge", "comparison")


@dataclass(frozen=True)
class IngestReport:
    questions_pooled: int
    paragraphs: int
    empty_paragraphs: int
    dropped_questions: int
    selected: dict[str, int]


def safe_filename(title: str) -> str:
    """ASCII-only, filesystem-safe name; hashed suffix when characters were lost."""
    normalised = unicodedata.normalize("NFC", title)
    ascii_title = (
        unicodedata.normalize("NFKD", normalised)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    cleaned = re.sub(r'[<>:"/\\|?*,()\'\[\]]', "_", ascii_title)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")[:100] or "untitled"
    if ascii_title != normalised or len(normalised) > 100:
        cleaned = f"{cleaned}_{hashlib.md5(normalised.encode('utf-8')).hexdigest()[:6]}"
    return cleaned


def norm_title(title: str) -> str:
    return unicodedata.normalize("NFC", title).strip()


def ingest(
    *,
    corpus_dir: str | Path,
    questions_dir: str | Path,
    pool: int = 300,
    evaluate: int = 100,
    seed: int = DEFAULT_SEED,
    clean: bool = False,
    smoke: bool = False,
) -> IngestReport:
    corpus = Path(corpus_dir)
    questions_out = Path(questions_dir)
    pool_n = 5 if smoke else pool
    eval_n = 4 if smoke else evaluate
    if clean and corpus.exists():
        shutil.rmtree(corpus)
    if not smoke:
        corpus.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(HF_DATASET, HF_CONFIG, split="validation", streaming=True)
    paragraphs: dict[str, str] = {}
    candidates: dict[str, list[QuestionRecord]] = {t: [] for t in QUESTION_TYPES}
    seen = empty = dropped = 0
    for row in dataset:
        if seen >= pool_n:
            break
        seen += 1
        row_files: dict[str, str] = {}
        for title, sentences in zip(
            row["context"]["title"], row["context"]["sentences"], strict=True
        ):
            text = " ".join(s.strip() for s in sentences if s.strip()).strip()
            if not text:
                empty += 1
                continue
            name = safe_filename(title)
            paragraphs[name] = text
            row_files[norm_title(title)] = name
        gold_titles = {norm_title(t) for t in row["supporting_facts"]["title"]}
        gold_files = sorted(
            {f"{row_files[t]}.txt" for t in gold_titles if t in row_files}
        )
        if len(gold_files) < 2:
            dropped += 1
            continue
        qtype = row.get("type", "bridge")
        if qtype not in candidates:
            continue
        candidates[qtype].append(
            QuestionRecord(
                id=str(row["id"]),
                question=row["question"],
                answer=row["answer"],
                gold_sources=tuple(gold_files),
                question_type=f"hotpot_{qtype}",
                construction_method="hotpotqa-native",
                evidence_requirement="text-multi-source",
                metadata={"level": row.get("level", "unknown")},
            )
        )

    rng = random.Random(seed)
    selected: dict[str, list[QuestionRecord]] = {}
    for qtype in QUESTION_TYPES:
        items = list(candidates[qtype])
        rng.shuffle(items)
        selected[qtype] = items[: eval_n // 2]

    if not smoke:
        for name, text in paragraphs.items():
            (corpus / f"{name}.txt").write_text(text, encoding="utf-8")
        for qtype, items in selected.items():
            write_questions(
                items,
                questions_out / f"questions_hotpotqa_{qtype}.json",
                source_file=f"{HF_DATASET}:{HF_CONFIG}:validation",
            )
    return IngestReport(
        questions_pooled=seen,
        paragraphs=len(paragraphs),
        empty_paragraphs=empty,
        dropped_questions=dropped,
        selected={qtype: len(items) for qtype, items in selected.items()},
    )
