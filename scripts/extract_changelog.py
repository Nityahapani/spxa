"""
Extracts the changelog entry for a given version from CHANGELOG.md.
Used by the release workflow to populate GitHub release notes.
Usage: python scripts/extract_changelog.py 0.1.0
"""

import re
import sys
from pathlib import Path


def extract(version: str) -> str:
    text = Path("CHANGELOG.md").read_text()
    pattern = rf"## {re.escape(version)}.*?\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return f"Release {version}"
    return match.group(1).strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: extract_changelog.py <version>")
        sys.exit(1)
    print(extract(sys.argv[1]))
