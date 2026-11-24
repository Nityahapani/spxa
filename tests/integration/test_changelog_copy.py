"""docs/changelog.md is the published copy of CHANGELOG.md and must not drift."""

from __future__ import annotations

import re
from pathlib import Path

import spxa

ROOT = Path(__file__).resolve().parents[2]


def test_docs_changelog_is_identical_to_the_root_changelog() -> None:
    root = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert (ROOT / "docs" / "changelog.md").read_text(encoding="utf-8") == root


def test_version_is_the_same_everywhere() -> None:
    v = spxa.__version__
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == v
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.findall(r"^\s*version: (\S+)$", cff, re.M) == [v, v]
    assert f"version = {{{v}}}" in (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    assert f"spxa v{v}." in api and f'spxa.__version__          # "{v}"' in api
    assert f"spxa v{v} " in (ROOT / "examples" / "full_showcase.py").read_text(encoding="utf-8")
    top = re.search(
        r"^## \[(?!Unreleased\])([^\]]+)\]",
        (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        re.M,
    )
    assert top.group(1) == v
