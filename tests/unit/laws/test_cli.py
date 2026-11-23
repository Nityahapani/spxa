"""spxa-laws: selection, formats, exit status."""

from __future__ import annotations

import json

import pytest

from spxa.laws.cli import main


def test_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list", "--tags", "axiom"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 5 and out[0].startswith("cf_at_zero")


def test_text_report_and_exit_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--names", "cf_at_zero,mean_is_additive", "-n", "5", "--seed", "3"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("spxa: 2 verified (2 laws, 5 examples each, seed 3")
    assert "verified      cf_at_zero" in captured.err


def test_json_and_markdown_to_file(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r.json"
    assert (
        main(["--names", "cf_is_bounded", "-n", "3", "--format", "json", "-o", str(out), "-q"]) == 0
    )
    data = json.loads(out.read_text())
    assert data["ok"] and data["results"][0]["law"] == "cf_is_bounded"
    assert capsys.readouterr().err == ""
    md = tmp_path / "r.md"
    assert (
        main(["--names", "cf_is_bounded", "-n", "3", "--format", "markdown", "-o", str(md), "-q"])
        == 0
    )
    assert "| `cf_is_bounded` | verified | 3 |" in md.read_text()


def test_bad_arguments_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--names", "no_such_law"]) == 2
    assert main(["--tags", "no_such_tag"]) == 2
    assert main(["-n", "0"]) == 2
    assert "no such laws" in capsys.readouterr().err


def test_failure_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    from spxa.laws import cli, expect, floats, law
    from spxa.laws.suite import Suite

    @law(x=floats(0.0, 1.0))
    def wrong(x):
        expect(x < 0.1, "no")

    monkeypatch.setattr(cli, "default_suite", lambda: Suite("t", wrong))
    assert main(["-n", "20", "-q"]) == 1
