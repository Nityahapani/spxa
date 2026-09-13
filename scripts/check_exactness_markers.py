"""
Scans tests/exactness/ and fails if any test function has a skip or xfail marker.
Called by the exactness-gate CI job.
"""

import ast
import sys
from pathlib import Path

FORBIDDEN_MARKERS = {"skip", "skipif", "xfail"}

def check_file(path: Path) -> list[str]:
    violations = []
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            name = None
            if isinstance(decorator, ast.Attribute):
                name = decorator.attr
            elif isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
            elif isinstance(decorator, ast.Name):
                name = decorator.id
            if name in FORBIDDEN_MARKERS:
                violations.append(
                    f"{path}:{node.lineno}: {node.name} has forbidden marker @pytest.mark.{name}"
                )
    return violations

def main() -> None:
    exactness_dir = Path("tests/exactness")
    if not exactness_dir.exists():
        print("tests/exactness/ does not exist yet — skipping check")
        sys.exit(0)

    all_violations: list[str] = []
    for test_file in exactness_dir.rglob("test_*.py"):
        all_violations.extend(check_file(test_file))

    if all_violations:
        print("Exactness test violations found:")
        for v in all_violations:
            print(f"  {v}")
        print(
            "\nExactness tests may never be skipped or xfailed. "
            "Open an issue with the label 'exactness-violation' if a test is mathematically wrong."
        )
        sys.exit(1)

    print(f"Checked {sum(1 for _ in exactness_dir.rglob('test_*.py'))} exactness test files — all clean.")

if __name__ == "__main__":
    main()
