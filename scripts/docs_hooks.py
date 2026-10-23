"""Expose canonical root policies to MkDocs without maintaining duplicate copies."""

from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files

ROOT = Path(__file__).resolve().parents[1]


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Build root policy pages directly from their canonical tracked sources."""
    for destination, source in (
        ("process_registry.md", "PROCESS_REGISTRY.md"),
        ("contributing.md", "CONTRIBUTING.md"),
        ("changelog.md", "CHANGELOG.md"),
    ):
        files.append(File.generated(config, destination, content=(ROOT / source).read_text()))
    return files
