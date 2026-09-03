"""Select which run files feed the tables: a frozen manifest or a directory scan.

A release uses a manifest, so every cell traces to an exact checksummed file.
A directory scan is for exploration only: it takes one newest valid run per
question set and generator as a unit and never backfills individual cells from
another timestamp.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..clients import DIAGNOSTICS, GENERATORS
from ..errors import ConfigurationError
from .manifest import load_manifest

MODEL_ALTERNATIVES = "|".join(re.escape(name) for name in GENERATORS + DIAGNOSTICS)
FILENAME_RE = re.compile(
    rf"^(summary|detail)_questions_(?P<qset>.+)_(?P<model>{MODEL_ALTERNATIVES})_(?P<stamp>\d{{8}}_\d{{6}})\.csv$"
)
QUESTION_PREFIX = "questions_"


def qset_name(question_set: str) -> str:
    """``questions_publaynet_text`` -> ``publaynet_text``."""
    return (
        question_set[len(QUESTION_PREFIX) :]
        if question_set.startswith(QUESTION_PREFIX)
        else question_set
    )


@dataclass(frozen=True)
class RunCell:
    summary: dict[str, str]
    detail_path: Path | None
    stamp: str

    @property
    def run_id(self) -> str:
        return self.summary.get("run_id", "")

    def detail_rows(self) -> list[dict[str, str]]:
        if self.detail_path is None:
            raise ConfigurationError(f"run {self.run_id} has no detail file")
        with self.detail_path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))


@dataclass
class RunSelection:
    provenance: str
    cells: dict[tuple[str, str, str], RunCell] = field(default_factory=dict)

    def get(self, qset: str, model: str, system: str) -> RunCell | None:
        return self.cells.get((qset, model, system))

    def question_sets(self) -> list[str]:
        return sorted({key[0] for key in self.cells})

    def coverage(self) -> dict[str, set[str]]:
        seen: dict[str, set[str]] = {}
        for qset, model, _ in self.cells:
            seen.setdefault(qset, set()).add(model)
        return seen

    def summary_rows(self) -> list[dict[str, str]]:
        return [cell.summary for cell in self.cells.values()]


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_is_valid(summary_rows: Iterable[dict[str, str]], qset: str) -> bool:
    """Wrong-corpus guard: a text-gold set whose baseline recall is zero is invalid."""
    if "figures" in qset:
        return True
    for row in summary_rows:
        if row.get("system") == "baseline":
            return float(row.get("recall", "0") or 0) > 0.0
    return True


def scan_runs(results_dir: str | Path) -> RunSelection:
    directory = Path(results_dir)
    if not directory.is_dir():
        raise ConfigurationError(f"results directory does not exist: {directory}")
    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for path in directory.rglob("*.csv"):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        kind = match.group(1)
        key = (match.group("qset"), match.group("model"))
        entry = grouped.setdefault(key, {}).setdefault(match.group("stamp"), {})
        entry["summary" if kind == "summary" else "detail"] = path
    selection = RunSelection(provenance=f"scan:{directory}")
    for (qset, model), by_stamp in grouped.items():
        for stamp in sorted(by_stamp, reverse=True):
            entry = by_stamp[stamp]
            summary_path = entry.get("summary")
            if summary_path is None:
                continue
            rows = _read_summary(summary_path)
            if not rows or not run_is_valid(rows, qset):
                continue
            for row in rows:
                selection.cells[(qset, model, row["system"])] = RunCell(
                    summary=row, detail_path=entry.get("detail"), stamp=stamp
                )
            break
    if not selection.cells:
        raise ConfigurationError(f"no usable run files under {directory}")
    return selection


def select_from_manifest(
    manifest_path: str | Path, root: str | Path = "."
) -> RunSelection:
    manifest = load_manifest(manifest_path)
    base = Path(root)
    selection = RunSelection(provenance=f"manifest:{manifest_path}")
    for run in manifest["runs"]:
        summary_path = base / run["summary"]
        detail_path = base / run["detail"]
        for row in _read_summary(summary_path):
            key = (qset_name(row["question_set"]), row["model"], row["system"])
            if key in selection.cells:
                raise ConfigurationError(
                    f"manifest names two runs for {key}; a release needs exactly one"
                )
            selection.cells[key] = RunCell(
                summary=row, detail_path=detail_path, stamp=run["run_id"]
            )
    return selection
