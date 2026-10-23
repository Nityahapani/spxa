"""Reject skip/xfail decorations, module markers and imperative skipping in exactness tests."""

import ast
import sys
from pathlib import Path

FORBIDDEN_MARKERS = {"skip", "skipif", "xfail", "importorskip"}


def check_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    aliases = set(FORBIDDEN_MARKERS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pytest":
            aliases.update(n.asname or n.name for n in node.names if n.name in FORBIDDEN_MARKERS)
    violations = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_MARKERS:
            violations.add((node.lineno, node.attr))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in aliases
        ):
            violations.add((node.lineno, node.func.id))
    return [
        f"{path}:{line}: forbidden exactness skip/xfail: {name}"
        for line, name in sorted(violations)
    ]


def main() -> None:
    files = sorted(Path("tests/exactness").rglob("test_*.py"))
    if not files:
        sys.exit("No exactness test files found")
    violations = [item for path in files for item in check_file(path)]
    if violations:
        sys.exit("\n".join(violations))
    print(f"Checked {len(files)} exactness test files; no skips or xfails")


if __name__ == "__main__":
    main()
