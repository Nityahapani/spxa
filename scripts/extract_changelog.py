"""Extract an actual version section; missing entries fail instead of fabricating notes."""

import re
import sys
from pathlib import Path


def extract(version: str, changelog: Path = Path("CHANGELOG.md")) -> str:
    text = changelog.read_text()
    version = version.removeprefix("v")
    heading = rf"^##\s+\[?v?{re.escape(version)}\]?(?:[ \t]+[^\n]*)?$"
    match = re.search(heading, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"No changelog section for {version}")
    rest = text[match.end() :]
    following = re.search(r"^##\s+", rest, re.MULTILINE)
    entry = rest[: following.start()] if following else rest
    if not entry.strip():
        raise ValueError(f"Empty changelog section for {version}")
    return entry.strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: extract_changelog.py <version>")
    try:
        print(extract(sys.argv[1]))
    except ValueError as error:
        sys.exit(str(error))
