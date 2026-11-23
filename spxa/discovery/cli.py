"""
Command line for process discovery.

Run it as ``python -m spxa.discovery`` (or the ``spxa-discover`` script).

From returns
::

    python -m spxa.discovery returns prices.csv --kind prices \\
        --families BM,VG,NIG,Gamma,NormalJumps --max-terms 2

reads the CSV (see :func:`~spxa.discovery.io.load_returns`), runs
:func:`~spxa.discovery.search.discover` and prints the report: the
headline sentence, the ranked table and any warnings.

From option quotes
::

    python -m spxa.discovery options quotes.csv --spot 100 --rate 0.02

reads a CSV with the columns ``strike``, ``maturity`` (years) and
``price``, plus an optional ``type`` column (``call``/``put``/``c``/``p``;
calls if omitted). It then runs
:func:`~spxa.discovery.options.discover_from_options`.

``--json`` prints :meth:`DiscoveryReport.to_dict` instead of the table.
The exit status is 0 on success and 2 on bad input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Optional, Sequence

import numpy as np

from spxa.discovery.grammar import Grammar, grammar_from_names
from spxa.discovery.io import load_returns
from spxa.discovery.options import OptionQuotes, discover_from_options
from spxa.discovery.report import DiscoveryReport
from spxa.discovery.search import discover


def _grammar(families: Optional[str], max_terms: int) -> Grammar:
    if not families:
        return Grammar(max_terms=max_terms)
    names = [n.strip() for n in families.split(",") if n.strip()]
    try:
        return grammar_from_names(names, max_terms=max_terms)
    except KeyError as exc:
        raise ValueError(f"unknown family {exc.args[0]!r}") from None


def read_option_quotes(
    path: str, spot: float, rate: float = 0.0, dividend: float = 0.0
) -> OptionQuotes:
    """Option quotes from a CSV with ``strike``, ``maturity``, ``price`` and optional ``type``."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [
            {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            for r in csv.DictReader(fh)
        ]
    if not rows:
        raise ValueError(f"{path}: no quotes")
    missing = {"strike", "maturity", "price"} - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(sorted(missing))}")

    def is_call(t: str) -> bool:
        t = t.lower()
        if t in ("", "c", "call"):
            return True
        if t in ("p", "put"):
            return False
        raise ValueError(f"{path}: option type must be call/put, got {t!r}")

    try:
        k = [float(r["strike"]) for r in rows]
        t = [float(r["maturity"]) for r in rows]
        p = [float(r["price"]) for r in rows]
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    c = [is_call(r.get("type", "")) for r in rows]
    return OptionQuotes.from_arrays(spot, k, t, p, np.array(c), rate, dividend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spxa-discover",
        description="Find the compositions of Lévy primitives that explain your data.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("csv", help="input CSV file")
        p.add_argument(
            "--families",
            help="comma-separated primitives (default: the full grammar); "
            "e.g. BM,VG,NIG,Gamma,NormalJumps",
        )
        p.add_argument("--max-terms", type=int, default=2, help="largest number of summands")
        p.add_argument("--criterion", choices=("bic", "aic"), default="bic")
        p.add_argument("--beam-width", type=int, default=3)
        p.add_argument("--exclude-degenerate", action="store_true")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--top", type=int, default=10, help="rows to print")
        p.add_argument("--json", action="store_true", help="print JSON instead of a table")

    r = sub.add_parser("returns", help="discover from a CSV of returns or prices")
    common(r)
    r.add_argument("--column", help="column name or 0-based index (default: auto)")
    r.add_argument("--kind", choices=("auto", "returns", "prices"), default="auto")
    r.add_argument("--simple", action="store_true", help="simple instead of log returns")
    r.add_argument(
        "--dt", type=float, help="sampling interval in years (default: inferred from dates, else 1)"
    )

    o = sub.add_parser("options", help="discover from a CSV of European option quotes")
    common(o)
    o.add_argument("--spot", type=float, required=True)
    o.add_argument("--rate", type=float, default=0.0)
    o.add_argument("--dividend", type=float, default=0.0)
    return parser


def run(args: argparse.Namespace) -> DiscoveryReport:
    grammar = _grammar(args.families, args.max_terms)
    if args.mode == "returns":
        column: Optional[str | int] = args.column
        if isinstance(column, str) and column.isdigit():
            column = int(column)
        series = load_returns(args.csv, column=column, kind=args.kind, log=not args.simple)
        return discover(
            series,
            dt=args.dt,
            grammar=grammar,
            criterion=args.criterion,
            beam_width=args.beam_width,
            seed=args.seed,
            exclude_degenerate=args.exclude_degenerate,
        )
    quotes = read_option_quotes(args.csv, args.spot, args.rate, args.dividend)
    return discover_from_options(
        quotes,
        grammar=grammar,
        criterion=args.criterion,
        beam_width=args.beam_width,
        seed=args.seed,
        exclude_degenerate=args.exclude_degenerate,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError) as exc:
        print(f"spxa-discover: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(top=args.top), indent=2, allow_nan=False))
    else:
        lines = [report.headline(), "", report.table(top=args.top)]
        lines += [f"warning: {w}" for w in report.warnings]
        print("\n".join(lines))
    return 0
