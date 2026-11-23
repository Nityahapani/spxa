"""End-to-end recovery: discover() finds the structure that generated the data.

Each case simulates increments exactly (see conftest) and searches a small
grammar. When two structures are close to equivalent, the test asserts what
can be identified and no more.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxa.discovery import discover, grammar_from_names

from .conftest import gamma_increments, merton_increments, vg_increments

DT = 1 / 252
GRAMMAR = grammar_from_names(["BM", "VG", "NormalJumps", "Gamma"], max_terms=2)

pytestmark = pytest.mark.slow


def _keys(report, top: int = 3) -> list[str]:
    return [c.structure_key for c in report.candidates[:top]]


def test_brownian_motion_is_not_overfitted() -> None:
    x = np.random.default_rng(4).normal(0.08 * DT, 0.2 * np.sqrt(DT), 3_000)
    r = discover(x, dt=DT, grammar=GRAMMAR, n_starts=2)
    assert r.best.structure_key == "BM + Drift"
    # the next structure needs BIC evidence it does not have
    assert r.candidates[1].bic - r.best.bic > 6
    (sigma,), (mu,) = r.best.model.params
    assert sigma == pytest.approx(0.2, rel=0.05)
    assert mu == pytest.approx(0.08, abs=3 * 0.2 / np.sqrt(3_000 * DT))


def test_merton_jump_diffusion_is_brownian_motion_plus_normal_jumps() -> None:
    x = merton_increments(0.05, 0.15, 10.0, -0.03, 0.04, DT, 5_000, seed=3)
    r = discover(x, dt=DT, grammar=GRAMMAR, n_starts=2)
    assert r.best.structure_key == "BM + NormalJumps + Drift"
    assert r.candidates[1].bic - r.best.bic > 6
    (sigma,), (lam, jm, js), _ = r.best.model.params
    assert sigma == pytest.approx(0.15, rel=0.1)
    assert 4.0 < lam < 20.0 and -0.06 < jm < 0.0 and 0.02 < js < 0.07
    assert r.best.g_pvalue > 0.05


def test_variance_gamma_beats_its_neighbours() -> None:
    x = vg_increments(0.2, 0.25, -0.3, 1 / 52, 2_000, seed=7, mu=0.1)
    g = grammar_from_names(["BM", "VG", "NIG", "NormalJumps"], max_terms=1)
    r = discover(x, dt=1 / 52, grammar=g, n_starts=2)
    assert r.best.structure_key == "VG + Drift"
    assert r.weights()[0] > 0.95


def test_sum_of_pure_jump_laws_up_to_near_equivalence() -> None:
    # VG + Gamma = (Γ⁺ + Γ₃) - Γ⁻ is close to a bilateral Gamma Γ⁺' - Γ⁻ with one
    # fewer parameter, so BIC may prefer either; both must outrank the rest and fit
    x = vg_increments(0.15, 0.2, -0.1, DT, 5_000, seed=1) + gamma_increments(
        4.0, 25.0, DT, 5_000, seed=20
    )
    r = discover(x, dt=DT, grammar=GRAMMAR, n_starts=2)
    top = set(_keys(r, 2))
    assert top == {"VG + Gamma + Drift", "-Gamma + Gamma + Drift"}
    assert r.best.g_pvalue > 0.05 and r.best.ks_distance < 0.01
    others = [c.bic for c in r.candidates[2:]]
    assert min(others) - r.best.bic > 10
