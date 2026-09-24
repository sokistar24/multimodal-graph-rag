"""Guards for things that have silently broken more than once.

The manuscript must never be committed. The `paper/` ignore rule was twice
stripped by an editor trimming the end of `.gitignore`, and each time the next
`git add -A` re-tracked the manuscript without anyone noticing. These tests
fail loudly instead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORE_FILE = REPO_ROOT / ".gitignore"


def ignore_rules() -> list[str]:
    lines = IGNORE_FILE.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"git unavailable or not a repository: {result.stderr.strip()}")
    return result.stdout


def test_manuscript_directory_is_ignored():
    assert "paper/" in ignore_rules(), (
        "the 'paper/' rule has gone missing from .gitignore; without it the next "
        "`git add -A` commits the manuscript"
    )


def test_the_ignore_rule_is_not_on_the_last_line():
    """It is stripped there. Keep something after it."""
    rules = [
        line.strip() for line in IGNORE_FILE.read_text(encoding="utf-8").splitlines()
    ]
    assert rules and rules[-1] != "paper/", (
        "'paper/' is the last line again, which is exactly where it keeps being "
        "trimmed away; move it up and keep other entries after it"
    )


def test_no_manuscript_file_is_tracked():
    tracked = [line for line in git("ls-files").splitlines() if line.strip()]
    offenders = [
        path
        for path in tracked
        if path.startswith("paper/")
        or Path(path).name in {"manuscript.tex", "manuscript.pdf", "references.bib"}
    ]
    assert not offenders, f"manuscript files are tracked: {offenders}"


def test_secrets_are_ignored():
    assert ".env" in ignore_rules()
    assert not [line for line in git("ls-files").splitlines() if line.strip() == ".env"]
