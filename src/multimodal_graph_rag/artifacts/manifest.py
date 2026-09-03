"""Release manifests that lock reported tables to exact, checksummed run files."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError

MANIFEST_SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> Mapping[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ConfigurationError(f"manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ConfigurationError("release manifest must use schema_version 1")
    if not manifest.get("runs"):
        raise ConfigurationError("release manifest must name at least one run")
    return manifest


def _relative_inside(base: Path, value: str) -> Path | None:
    candidate = (base / value).resolve()
    try:
        return candidate.relative_to(base.resolve())
    except ValueError:
        return None


def validate_manifest(path: str | Path, root: str | Path = ".") -> list[str]:
    """Every problem that makes a manifest unfit for a release, as messages."""
    manifest = load_manifest(path)
    base = Path(root)
    errors: list[str] = []
    seen_ids: set[str] = set()
    question_set = manifest.get("question_set")
    if question_set:
        question_path = base / question_set
        if not question_path.is_file():
            errors.append(f"missing question_set: {question_set}")
        elif not manifest.get("question_set_sha256"):
            errors.append("question_set_sha256 is required")
        elif file_sha256(question_path) != manifest["question_set_sha256"]:
            errors.append(f"question_set checksum mismatch: {question_set}")
    for run in manifest["runs"]:
        run_id = run.get("run_id")
        if not run_id or run_id in seen_ids:
            errors.append(f"missing or duplicate run_id: {run_id!r}")
        seen_ids.add(run_id)
        for key in ("summary", "detail"):
            value = run.get(key)
            if not value:
                errors.append(f"run {run_id!r} has no {key} path")
                continue
            relative = _relative_inside(base, value)
            if relative is None:
                errors.append(f"run {run_id!r} {key} escapes repository: {value}")
                continue
            if relative.parts and relative.parts[0].lower() == "archive":
                errors.append(f"run {run_id!r} {key} points into archive: {value}")
            candidate = base / value
            if not candidate.is_file():
                errors.append(f"run {run_id!r} missing {key}: {value}")
            expected_hash = run.get(f"{key}_sha256")
            if not expected_hash:
                errors.append(f"run {run_id!r} has no {key}_sha256")
            elif candidate.is_file() and file_sha256(candidate) != expected_hash:
                errors.append(f"run {run_id!r} {key} checksum mismatch: {value}")
        summary = base / str(run.get("summary", ""))
        if summary.is_file():
            with summary.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                errors.append(f"run {run_id!r} has empty summary")
            for row in rows:
                if row.get("run_id") and row["run_id"] != run_id:
                    errors.append(f"run id mismatch in {run.get('summary')}")
                if (
                    row.get("config_hash")
                    and run.get("config_hash") != row["config_hash"]
                ):
                    errors.append(f"config hash mismatch in {run.get('summary')}")
                if int(float(row.get("n_errors", "0") or 0)):
                    errors.append(f"run {run_id!r} contains generation errors")
    for artifact in manifest.get("artifacts", []):
        value = artifact.get("path")
        expected_hash = artifact.get("sha256")
        candidate = base / str(value or "")
        if not value or not candidate.is_file():
            errors.append(f"missing release artifact: {value!r}")
        elif not expected_hash or file_sha256(candidate) != expected_hash:
            errors.append(f"release artifact checksum mismatch: {value}")
    return errors


def load_frozen_summary_rows(
    manifest_path: str | Path, root: str | Path = "."
) -> list[dict[str, str]]:
    manifest = load_manifest(manifest_path)
    base = Path(root)
    rows = []
    for run in manifest["runs"]:
        with (base / run["summary"]).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("run_id") and row["run_id"] != run["run_id"]:
                    raise ConfigurationError(f"run id mismatch in {run['summary']}")
                row["_manifest_run_id"] = run["run_id"]
                row["_source"] = run["summary"]
                rows.append(row)
    return rows


def write_matrix(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    materialised = list(rows)
    if not materialised:
        raise ConfigurationError("cannot write an empty artifact matrix")
    fieldnames = sorted({key for row in materialised for key in row})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialised)


def freeze_manifest(
    summary_paths: Sequence[str | Path],
    *,
    root: str | Path = ".",
    release_id: str,
    commit: str,
    question_set: str | None = None,
) -> dict[str, Any]:
    """Build a manifest from summary files, pairing each with its detail file."""
    base = Path(root)
    runs = []
    for summary in summary_paths:
        summary_path = base / summary
        if not summary_path.is_file():
            raise ConfigurationError(f"summary does not exist: {summary_path}")
        detail_path = summary_path.with_name(
            summary_path.name.replace("summary_", "detail_", 1)
        )
        if not detail_path.is_file():
            raise ConfigurationError(f"detail file does not exist: {detail_path}")
        with summary_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ConfigurationError(f"summary is empty: {summary_path}")
        run_ids = {row.get("run_id", "") for row in rows}
        hashes = {row.get("config_hash", "") for row in rows}
        if len(run_ids) != 1 or len(hashes) != 1:
            raise ConfigurationError(f"summary mixes runs: {summary_path}")
        runs.append(
            {
                "run_id": run_ids.pop(),
                "config_hash": hashes.pop(),
                "summary": summary_path.relative_to(base).as_posix(),
                "summary_sha256": file_sha256(summary_path),
                "detail": detail_path.relative_to(base).as_posix(),
                "detail_sha256": file_sha256(detail_path),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "release_id": release_id,
        "commit": commit,
        "runs": runs,
    }
    if question_set:
        manifest["question_set"] = question_set
        manifest["question_set_sha256"] = file_sha256(base / question_set)
    return manifest
