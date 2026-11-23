"""Reading returns and prices from CSV files."""

from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pytest

from spxa.discovery.io import infer_dt, load_returns


def _business_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_prices_with_dates_give_daily_log_returns(tmp_path) -> None:
    days = _business_days(date(2024, 1, 2), 30)
    prices = 100 * np.exp(np.cumsum(np.r_[0.0, np.linspace(-0.01, 0.01, 29)]))
    lines = ["Date,Open,Adj Close,Volume"] + [
        f"{d.isoformat()},1.0,{p:.10f},1000" for d, p in zip(days, prices)
    ]
    path = tmp_path / "px.csv"
    path.write_text("\n".join(lines))
    s = load_returns(path)
    assert s.column == "Adj Close" and s.kind == "prices"
    np.testing.assert_allclose(s.returns, np.diff(np.log(prices)), atol=1e-10)
    assert s.dt == pytest.approx(1 / 252)
    assert s.dates[0] == days[1] and len(s) == 29
    simple = load_returns(path, log=False)
    np.testing.assert_allclose(simple.returns, prices[1:] / prices[:-1] - 1, atol=1e-10)


def test_newest_first_files_are_reversed() -> None:
    days = _business_days(date(2024, 3, 1), 5)
    prices = [10.0, 11.0, 12.1, 12.1, 13.31]
    rows = [f"{d.isoformat()},{p}" for d, p in zip(days, prices)][::-1]
    s = load_returns(io.StringIO("date,close\n" + "\n".join(rows)))
    np.testing.assert_allclose(s.returns, np.log([1.1, 1.1, 1.0, 1.1]))
    assert list(s.dates) == days[1:]


def test_return_columns_and_missing_values() -> None:
    text = "when;ret;note\n2024-01-05;0.01;a\n2024-01-12;NA;b\n2024-01-19;-0.5%;c\n2024-01-26;0.002;d\n"
    s = load_returns(io.StringIO(text))
    assert s.kind == "returns" and s.column == "ret"
    np.testing.assert_allclose(s.returns, [0.01, -0.005, 0.002])
    assert s.dt == pytest.approx(1 / 52)


def test_column_selection_and_auto_kind() -> None:
    text = "a,b\n1,101.0\n2,102.5\n3,101.7\n4,103.2\n"
    s = load_returns(io.StringIO(text))
    # unnamed, positive and large: prices
    assert s.column == "b" and s.kind == "prices" and s.dt is None
    by_index = load_returns(io.StringIO(text), column=1, kind="returns")
    np.testing.assert_allclose(by_index.returns, [101.0, 102.5, 101.7, 103.2])
    by_name = load_returns(io.StringIO(text), column="A", kind="returns")
    np.testing.assert_allclose(by_name.returns, [1, 2, 3, 4])
    small = load_returns(io.StringIO("x\n0.01\n-0.02\n0.005\n"))
    assert small.kind == "returns"


def test_errors() -> None:
    with pytest.raises(ValueError, match="no column named"):
        load_returns(io.StringIO("a\n1\n2\n"), column="b")
    with pytest.raises(ValueError, match="out of range"):
        load_returns(io.StringIO("a\n1\n2\n"), column=3)
    with pytest.raises(ValueError, match="strictly positive"):
        load_returns(io.StringIO("close\n1\n-2\n3\n"))
    with pytest.raises(ValueError, match="header"):
        load_returns(io.StringIO("close\n"))
    with pytest.raises(ValueError, match="no numeric"):
        load_returns(io.StringIO("a,b\nx,y\nz,w\n"))


def test_infer_dt() -> None:
    start = date(2023, 1, 1)
    calendar_daily = [start + timedelta(days=i) for i in range(30)]
    assert infer_dt(calendar_daily) == pytest.approx(1 / 365.25)
    monthly = [date(2023, m, 1) for m in range(1, 13)]
    assert infer_dt(monthly) == pytest.approx(1 / 12)
    assert infer_dt([start, start]) is None
    with pytest.raises(ValueError):
        infer_dt([start, start, start])
