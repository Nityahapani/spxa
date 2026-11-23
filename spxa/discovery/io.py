r"""
Reading return series from CSV files.

:func:`load_returns` turns a CSV file of returns, or of prices, into the
increments that :func:`~spxa.discovery.search.discover` fits, together with
the sampling interval. It uses only the standard library, so pandas is not
needed.

Column choice
    ``column=`` takes a header name or a 0-based index. Without it, the loader
    prefers a header that looks like returns (``return``, ``ret``,
    ``log_return``), then one that looks like prices (``adj close``,
    ``close``, ``price``). Failing both, it takes the last numeric column.

Returns or prices
    ``kind="auto"`` decides from the header name. If the name settles
    nothing, a strictly positive column with median absolute value above
    0.5 is treated as prices: no plausible per-period return series looks
    like that. Prices :math:`P_t` become log returns :math:`\log(P_{t+1}/P_t)`,
    or simple returns with ``log=False``.

Sampling interval
    With an ISO date column (``YYYY-MM-DD``), the median spacing between
    consecutive dates sets ``dt`` in years:

    - spacing of 1 to 4 calendar days, from data with no weekend dates,
      gives :math:`1/252` (trading days);
    - otherwise a median of :math:`d` days gives :math:`d/7 \cdot 1/52` for
      weekly data, :math:`1/12` for 28 to 31 days, and :math:`d/365.25` in
      general.

    Without dates ``dt`` is ``None`` and the caller must supply it.
"""

from __future__ import annotations

import csv
import io
import math
import os
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional, Sequence, Union

import numpy as np

_MISSING = {"", "na", "nan", "null", "none", ".", "-", "n/a", "#n/a"}
_RETURN_NAMES = ("log_return", "logret", "log return", "returns", "return", "ret", "r")
_PRICE_NAMES = ("adj close", "adj_close", "adjclose", "close", "price", "last", "settle")


@dataclass(frozen=True)
class ReturnSeries:
    """Increments read from a file.

    Attributes
    ----------
    returns :
        The increments (log returns unless ``log=False`` was used on prices).
    dt :
        Sampling interval in years inferred from dates, or ``None``.
    column :
        Header of the column that was read.
    kind :
        ``"returns"`` or ``"prices"``: what the column contained.
    dates :
        ISO dates of the increments (the later date of each pair for prices),
        if a date column was found.
    """

    returns: np.ndarray
    dt: Optional[float]
    column: str
    kind: str
    dates: Optional[tuple[date, ...]] = None

    def __len__(self) -> int:
        return int(self.returns.size)


def _parse_float(text: str) -> Optional[float]:
    t = text.strip().replace(",", "")
    if t.lower() in _MISSING:
        return None
    pct = t.endswith("%")
    try:
        v = float(t[:-1] if pct else t)
    except ValueError:
        return None
    return v / 100.0 if pct else v


def _parse_date(text: str) -> Optional[date]:
    t = text.strip()[:10]
    try:
        return date.fromisoformat(t)
    except ValueError:
        return None


def _match(header: str, names: Sequence[str]) -> bool:
    h = header.strip().lower()
    return h in names or any(h.startswith(n + " ") or h.startswith(n + "_") for n in names)


def infer_dt(dates: Sequence[date]) -> Optional[float]:
    """Sampling interval in years from a sorted sequence of dates (see the module docstring)."""
    if len(dates) < 3:
        return None
    gaps = np.diff([d.toordinal() for d in dates]).astype(float)
    if np.any(gaps <= 0):
        raise ValueError("dates must be strictly increasing")
    med = float(np.median(gaps))
    weekends = any(d.weekday() >= 5 for d in dates)
    if med <= 4 and not weekends:
        return 1.0 / 252.0
    if 6 <= med <= 8:
        return (med / 7.0) / 52.0
    if 28 <= med <= 31:
        return 1.0 / 12.0
    return med / 365.25


def load_returns(
    source: Union[str, os.PathLike[str], io.TextIOBase],
    column: Union[str, int, None] = None,
    kind: Literal["auto", "returns", "prices"] = "auto",
    log: bool = True,
    delimiter: Optional[str] = None,
) -> ReturnSeries:
    """Read increments from a CSV file of returns or prices.

    Parameters
    ----------
    source :
        A path or an open text file. The first row must be a header.
    column :
        Header name (case-insensitive) or 0-based index of the data column.
    kind :
        ``"returns"``, ``"prices"`` or ``"auto"`` (see the module docstring).
    log :
        For prices, log returns (default) or simple returns.
    delimiter :
        Field separator; sniffed from the header when omitted.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, newline="", encoding="utf-8-sig") as fh:
            text = fh.read()
    else:
        text = source.read()
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("the file needs a header row and at least one data row")
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(lines[0], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    rows = [r for r in csv.reader(lines, delimiter=delimiter) if any(c.strip() for c in r)]
    header, body = [h.strip() for h in rows[0]], rows[1:]
    ncol = len(header)

    def cells(j: int) -> list[str]:
        return [r[j] if j < len(r) else "" for r in body]

    numeric = [
        j
        for j in range(ncol)
        if sum(_parse_float(c) is not None for c in cells(j)) >= max(2, len(body) // 2)
    ]
    date_col = next(
        (
            j
            for j in range(ncol)
            if j not in numeric
            and sum(_parse_date(c) is not None for c in cells(j)) >= max(2, len(body) // 2)
        ),
        None,
    )

    if column is None:
        col = next((j for j in numeric if _match(header[j], _RETURN_NAMES)), None)
        if col is None:
            col = next((j for j in numeric if _match(header[j], _PRICE_NAMES)), None)
        if col is None:
            if not numeric:
                raise ValueError("no numeric column found")
            col = numeric[-1]
    elif isinstance(column, int):
        if not 0 <= column < ncol:
            raise ValueError(f"column index {column} out of range for {ncol} columns")
        col = column
    else:
        lower = [h.lower() for h in header]
        if column.strip().lower() not in lower:
            raise ValueError(f"no column named {column!r}; the header is {header}")
        col = lower.index(column.strip().lower())

    values: list[float] = []
    stamps: list[Optional[date]] = []
    for r in body:
        v = _parse_float(r[col]) if col < len(r) else None
        if v is None or not math.isfinite(v):
            continue
        values.append(v)
        stamps.append(
            _parse_date(r[date_col]) if date_col is not None and date_col < len(r) else None
        )
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError(f"column {header[col]!r} has fewer than two numeric values")

    dates: Optional[list[date]] = None
    if date_col is not None and all(s is not None for s in stamps):
        dates = [s for s in stamps if s is not None]
        if len(dates) >= 2 and dates[0] > dates[-1]:  # newest first, as many vendors export
            dates.reverse()
            x = x[::-1].copy()

    # the spacing of the observation dates, taken over every dated row so that
    # a missing value does not look like a longer sampling interval
    dt = None
    if date_col is not None:
        dt = infer_dt(sorted({d for d in map(_parse_date, cells(date_col)) if d is not None}))

    if kind == "auto":
        if _match(header[col], _RETURN_NAMES):
            kind = "returns"
        elif _match(header[col], _PRICE_NAMES):
            kind = "prices"
        else:
            kind = "prices" if np.all(x > 0) and np.median(np.abs(x)) > 0.5 else "returns"
    if kind == "prices":
        if np.any(x <= 0):
            raise ValueError("prices must be strictly positive")
        x = np.diff(np.log(x)) if log else x[1:] / x[:-1] - 1.0
        if dates is not None:
            dates = dates[1:]
    elif kind != "returns":
        raise ValueError(f"kind must be 'auto', 'returns' or 'prices', got {kind!r}")

    return ReturnSeries(
        returns=x,
        dt=dt,
        column=header[col],
        kind=kind,
        dates=tuple(dates) if dates is not None else None,
    )
