"""Closed-form primitive families against Cauchy integrals and the zoo."""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest

from spxa.discovery.families import FAMILIES, get_family

# one generic member of every family (natural parameters)
MEMBERS = {
    "Drift": (0.3,),
    "BM": (0.25,),
    "Gamma": (2.5, 4.0),
    "IG": (0.8, 1.7),
}


def test_every_family_has_a_member() -> None:
    assert set(MEMBERS) == set(FAMILIES)


def cauchy_cumulants(name: str, p: tuple[float, ...], order: int) -> np.ndarray:
    """κ_n = n! [z^n] K(z) by the trapezoidal rule on a circle inside the strip."""
    fam = get_family(name)
    lo, hi = fam.strip(p)
    r = 0.5 * min(-lo, hi, 2.0)
    m = 256
    z = r * np.exp(2j * np.pi * np.arange(m) / m)
    vals = fam.log_mgf(z, p)
    coef = np.fft.fft(vals) / m
    return np.array([(coef[n] / r**n).real * factorial(n) for n in range(1, order + 1)])


@pytest.mark.parametrize("name", sorted(MEMBERS))
def test_closed_form_cumulants_match_the_cumulant_exponent(name: str) -> None:
    p = MEMBERS[name]
    fam = get_family(name)
    exact = fam.cumulants(p, 8)
    numeric = cauchy_cumulants(name, p, 8)
    scale = np.abs(exact) + np.abs(numeric).max() * 1e-12 + 1e-14
    assert np.all(np.abs(exact - numeric) <= 1e-8 * scale + 1e-12), (exact, numeric)


ZOO_ORDER = {"VG": 4}


@pytest.mark.parametrize("name", sorted(MEMBERS))
def test_cumulants_and_char_func_match_the_zoo_process(name: str) -> None:
    p = MEMBERS[name]
    fam = get_family(name)
    proc = fam.to_process(p)
    order = ZOO_ORDER.get(name, 4)
    zoo = proc.cumulants(order)
    np.testing.assert_allclose(
        [float(zoo[n]) for n in range(1, order + 1)], fam.cumulants(p, order), rtol=1e-7, atol=1e-12
    )
    u = np.array([-3.0, -0.5, 0.7, 2.0])
    for t in (0.5, 2.0):
        cf = np.array([complex(proc.char_func(float(x), t=t)) for x in u])
        np.testing.assert_allclose(cf, np.exp(t * fam.char_exponent(u, p)), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", sorted(MEMBERS))
def test_free_parameter_round_trip(name: str) -> None:
    fam = get_family(name)
    p = MEMBERS[name]
    np.testing.assert_allclose(fam.from_free(fam.to_free(p)), p, rtol=1e-12, atol=1e-14)
    # every point of R^d maps into the domain
    rng = np.random.default_rng(3)
    for _ in range(20):
        fam.from_free(rng.normal(scale=2.0, size=fam.n_params))


@pytest.mark.parametrize("name", sorted(MEMBERS))
def test_rescale_and_unit_scale(name: str) -> None:
    fam = get_family(name)
    p = MEMBERS[name]
    k = fam.cumulants(p, 6)
    c = 1.7
    powers = c ** np.arange(1, 7)
    np.testing.assert_allclose(
        fam.cumulants(fam.rescale(p, c), 6), powers * k, rtol=1e-11, atol=1e-14
    )
    s, unit = fam.unit_scale(p)
    np.testing.assert_allclose(fam.cumulants(fam.rescale(unit, s), 6), k, rtol=1e-11, atol=1e-14)
    with pytest.raises(ValueError):
        fam.rescale(p, -1.0)


@pytest.mark.parametrize("name", sorted(set(MEMBERS) - {"Drift"}))
def test_typical_member_has_the_requested_spread(name: str) -> None:
    fam = get_family(name)
    for s in (0.01, 0.3):
        k = fam.cumulants(fam.typical(s), 2)
        assert k[1] == pytest.approx(s**2, rel=1e-9) or name in ("Gamma", "IG", "Poisson")
        if name in ("Gamma", "IG", "Poisson"):
            assert np.sqrt(k[1]) == pytest.approx(s, rel=1e-9)


@pytest.mark.parametrize("name", ["Gamma", "IG"])
def test_strip_edges(name: str) -> None:
    fam = get_family(name)
    p = MEMBERS[name]
    lo, hi = fam.strip(p)
    assert lo < 0 < hi
    inside = np.array([0.99 * hi] + ([0.99 * lo] if np.isfinite(lo) else []))
    assert np.all(np.isfinite(fam.log_mgf(inside, p).real))
    # the moment generating function blows up (or the formula leaves the real line) at the edge
    k_hi = fam.log_mgf(np.array([hi * (1 - 1e-9)]), p).real[0]
    assert k_hi > fam.log_mgf(np.array([0.9 * hi]), p).real[0]


def test_subordinator_flags_and_domains() -> None:
    subs = {n for n, f in FAMILIES.items() if f.is_subordinator}
    assert subs == {"Gamma", "IG"}
    for name, bad in [
        ("BM", (-1.0,)),
    ]:
        with pytest.raises(ValueError):
            get_family(name).cumulants(bad, 2)
    with pytest.raises(KeyError):
        get_family("Heston")
    assert get_family("Gamma").format(MEMBERS["Gamma"], digits=3) == "GammaProcess(a=2.5, b=4)"
