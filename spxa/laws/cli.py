"""
Command line for the law suite.

Run it as ``python -m spxa.laws`` (or the ``spxa-laws`` script)::

    python -m spxa.laws                          # every library law, 100 examples
    python -m spxa.laws --tags algebra,subordination --examples 300 --seed 7
    python -m spxa.laws --exclude-tags simulation --format markdown -o laws.md
    python -m spxa.laws --list

It checks :func:`~spxa.laws.suite.default_suite`, filtered by ``--names``,
``--tags`` and ``--exclude-tags``, and prints the report as a table, as
Markdown or as JSON (``--format``). One line per law is printed to stderr as
results arrive (``--quiet`` turns it off). The exit status is 0 when every
law is verified, 1 when any is not, and 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from spxa.laws.law import LawResult
from spxa.laws.suite import default_suite


def _csv(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spxa-laws",
        description="Search the spxa law library for counterexamples.",
    )
    p.add_argument("--examples", "-n", type=int, default=100, help="examples per law (100)")
    p.add_argument("--seed", type=int, default=0, help="random seed (0)")
    p.add_argument("--names", type=_csv, help="comma-separated law names")
    p.add_argument("--tags", type=_csv, help="keep laws with any of these tags")
    p.add_argument("--exclude-tags", type=_csv, default=[], help="drop laws with these tags")
    p.add_argument(
        "--format", choices=["text", "markdown", "json"], default="text", help="report format"
    )
    p.add_argument("--output", "-o", help="write the report here instead of stdout")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first failure")
    p.add_argument("--no-shrink", action="store_true", help="report counterexamples unshrunk")
    p.add_argument("--quiet", "-q", action="store_true", help="no progress lines on stderr")
    p.add_argument("--list", action="store_true", help="list the selected laws and exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.examples < 1:
        print("spxa-laws: --examples must be at least 1", file=sys.stderr)
        return 2
    try:
        suite = default_suite().select(
            names=args.names, tags=args.tags, exclude_tags=args.exclude_tags
        )
    except KeyError as exc:
        print(f"spxa-laws: {exc.args[0]}", file=sys.stderr)
        return 2
    if not len(suite):
        print("spxa-laws: no laws selected", file=sys.stderr)
        return 2
    if args.list:
        for law_ in suite:
            print(f"{law_.name:<36} [{', '.join(law_.tags)}]  {law_.description}")
        return 0

    def progress(r: LawResult) -> None:
        if not args.quiet:
            print(f"  {r.status:<13} {r.law} ({r.elapsed:.1f} s)", file=sys.stderr, flush=True)

    report = suite.check(
        n_examples=args.examples,
        seed=args.seed,
        progress=progress,
        fail_fast=args.fail_fast,
        shrink=not args.no_shrink,
    )
    if args.format == "json":
        text = json.dumps(report.to_dict(), indent=2)
    elif args.format == "markdown":
        text = report.to_markdown()
    else:
        text = str(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0 if report.ok else 1
