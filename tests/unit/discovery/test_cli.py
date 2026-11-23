"""The spxa-discover command line."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from spxa.discovery import VARIANCE_GAMMA, Model, leaf
from spxa.discovery.cli import main, read_option_quotes
from spxa.discovery.options import OptionQuotes, model_prices

from .conftest import vg_increments


@pytest.fixture(scope="module")
def returns_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    x = vg_increments(0.2, 0.25, -0.3, 1 / 52, 1500, seed=3, mu=0.1)
    path = tmp_path_factory.mktemp("cli") / "returns.csv"
    path.write_text("ret\n" + "\n".join(f"{v:.17g}" for v in x) + "\n")
    return path


@pytest.fixture(scope="module")
def quotes_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    K = np.tile([85.0, 95.0, 100.0, 105.0, 115.0], 2)
    T = np.repeat([0.25, 1.0], 5)
    call = K >= 100
    shell = OptionQuotes.from_arrays(100.0, K, T, np.ones_like(K), call, rate=0.03)
    p = model_prices(Model(leaf(VARIANCE_GAMMA), [(0.15, 0.3, -0.2)]), shell, n_terms=4096)
    rows = [f"{k:g},{t:g},{v:.17g},{'C' if c else 'put'}" for k, t, v, c in zip(K, T, p, call)]
    path = tmp_path_factory.mktemp("cli") / "quotes.csv"
    path.write_text("Strike,Maturity,Price,Type\n" + "\n".join(rows) + "\n")
    return path


def test_returns_table(returns_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "returns",
            str(returns_csv),
            "--families",
            "BM,VG",
            "--max-terms",
            "1",
            "--dt",
            str(1 / 52),
        ]
    )
    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert out[0].startswith("VarianceGamma(") and "matches your data within" in out[0]
    assert "ΔBIC" in out[2] and len([ln for ln in out if ln[:3].strip().isdigit()]) == 2


def test_returns_json(returns_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["returns", str(returns_csv), "--families", "BM,VG", "--max-terms", "1", "--json"]
    assert main(args + ["--dt", str(1 / 52)]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["method"] == "likelihood" and d["n_obs"] == 1500
    assert d["candidates"][0]["structure"] == "VG + Drift"
    assert d["candidates"][0]["parameters"]["VG.nu"] == pytest.approx(0.25, abs=0.06)


def test_option_quotes(quotes_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    q = read_option_quotes(str(quotes_csv), 100.0, 0.03)
    assert len(q) == 10 and q.call.tolist() == [False, False, True, True, True] * 2
    args = ["options", str(quotes_csv), "--spot", "100", "--rate", "0.03"]
    assert main(args + ["--families", "BM,VG", "--max-terms", "1", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["method"] == "options" and d["candidates"][0]["structure"] == "VG"
    p = d["candidates"][0]["parameters"]
    assert (p["VG.sigma"], p["VG.nu"], p["VG.theta"]) == pytest.approx((0.15, 0.3, -0.2), rel=1e-3)


def test_errors(tmp_path: Path, returns_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["returns", str(tmp_path / "missing.csv")]) == 2
    assert main(["returns", str(returns_csv), "--families", "BM,Wiener"]) == 2
    assert "unknown family 'Wiener'" in capsys.readouterr().err
    bad = tmp_path / "bad.csv"
    bad.write_text("strike,price\n100,5\n")
    assert main(["options", str(bad), "--spot", "100"]) == 2
    assert "missing column(s) maturity" in capsys.readouterr().err
    bad.write_text("strike,maturity,price,type\n100,1,5,straddle\n")
    assert main(["options", str(bad), "--spot", "100"]) == 2
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_module_entry_point(returns_csv: Path) -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "spxa.discovery",
            "returns",
            str(returns_csv),
            "--families",
            "BM",
            "--max-terms",
            "1",
            "--dt",
            "0.0192307692",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.startswith("BrownianMotion(")
