"""Exact parameter derivatives of characteristic exponents and cumulants (DiffProcess)."""

from __future__ import annotations

from typing import Any, Callable, List

import numpy as np
import pytest

from spxa.graph.autodiff.families import RULES
from spxa.graph.autodiff.process import (
    DiffProcess,
    Param,
    UnsupportedProcessError,
    check_against_zoo,
)
from spxa.zoo.levy.brownian import BrownianMotion
from spxa.zoo.levy.cgmy import CGMY
from spxa.zoo.levy.drift import Drift
from spxa.zoo.levy.gamma import GammaProcess
from spxa.zoo.levy.gh import GeneralizedHyperbolicProcess
from spxa.zoo.levy.inverse_gaussian import InverseGaussianProcess
from spxa.zoo.levy.kou import KouJumpDiffusion
from spxa.zoo.levy.meixner import MeixnerProcess
from spxa.zoo.levy.merton import MertonJumpDiffusion
from spxa.zoo.levy.negative_binomial import NegativeBinomialProcess
from spxa.zoo.levy.nig import NIG
from spxa.zoo.levy.nts import NormalTemperedStableProcess
from spxa.zoo.levy.poisson import PoissonProcess
from spxa.zoo.levy.stable import AlphaStable
from spxa.zoo.levy.tempered_stable import TemperedStable
from spxa.zoo.levy.vg import VarianceGamma

mp = pytest.importorskip("mpmath")


@pytest.fixture(autouse=True)
def _restore_mp_precision() -> Any:
    dps = mp.mp.dps
    yield
    mp.mp.dps = dps


U = np.array([-6.0, -2.5, -0.7, 0.0, 0.3, 1.1, 4.0, 9.0])


def _families() -> List[Any]:
    return [
        BrownianMotion(0.1, 0.3),
        Drift(0.2),
        GammaProcess(1.3, 2.0),
        InverseGaussianProcess(0.8, 1.5),
        VarianceGamma(0.2, 0.3, -0.1),
        NIG(3.0, -0.5, 0.7, 0.05),
        MeixnerProcess(0.4, -0.6, 1.2, 0.02),
        TemperedStable(0.6, 1.1, 2.0),
        NormalTemperedStableProcess(0.5, 1.2, 0.9, -0.1, 0.25),
        CGMY(1.0, 5.0, 8.0, 0.7),
        CGMY(1.0, 5.0, 8.0, 1.5),
        CGMY(1.0, 5.0, 8.0, 0.0),
        CGMY(1.0, 5.0, 8.0, 1.0),
        CGMY(1.0, 5.0, 8.0, 0.03),
        CGMY(1.0, 5.0, 8.0, 0.97),
        PoissonProcess(2.0, 0.5),
        MertonJumpDiffusion(0.05, 0.2, 1.0, -0.05, 0.15),
        KouJumpDiffusion(0.05, 0.2, 1.0, 0.4, 10.0, 5.0),
        NegativeBinomialProcess(1.5, 0.3),
        GeneralizedHyperbolicProcess(1.2, 3.0, -0.5, 0.8, 0.03),
        GeneralizedHyperbolicProcess(-0.5, 2.0, 0.4, 1.5, 0.0),
    ]


def _ids(ps: List[Any]) -> List[str]:
    return [f"{type(p).__name__}-{i}" for i, p in enumerate(ps)]


FAMS = _families()


def _richardson(f: Callable[[np.ndarray], Any], x: np.ndarray, k: int, h: float) -> Any:
    """Fourth-order central difference with one Richardson step (error O(h^6))."""

    def d(hh: float) -> Any:
        e = np.zeros_like(x)
        e[k] = hh
        return (
            -np.asarray(f(x + 2 * e))
            + 8 * np.asarray(f(x + e))
            - 8 * np.asarray(f(x - e))
            + np.asarray(f(x - 2 * e))
        ) / (12 * hh)

    return (64 * d(h / 2) - d(h)) / 63


# ── value agreement with the zoo ─────────────────────────────────────────────


def test_every_rule_has_a_zoo_test() -> None:
    covered = {type(p).__name__ for p in FAMS} | {"AlphaStable"}
    assert set(RULES) == covered


@pytest.mark.parametrize("proc", FAMS, ids=_ids(FAMS))
def test_exponent_matches_zoo(proc: Any) -> None:
    assert check_against_zoo(DiffProcess(proc), U) < 1e-12


@pytest.mark.parametrize("proc", FAMS, ids=_ids(FAMS))
def test_cumulants_match_zoo(proc: Any) -> None:
    ours = DiffProcess(proc).cumulants(4)
    c = proc.cumulants(4)
    theirs = np.array([c[n] for n in range(1, 5)])
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-13)


# ── exact gradients and Hessians ────────────────────────────────────────────


@pytest.mark.parametrize("proc", FAMS, ids=_ids(FAMS))
def test_gradient_matches_richardson(proc: Any) -> None:
    dp = DiffProcess(proc)
    th = dp.values
    psi, g = dp.char_exponent_and_grad(U)
    np.testing.assert_allclose(psi, dp.char_exponent(U), rtol=1e-14, atol=1e-16)
    for k in range(len(th)):
        h = 1e-3 * max(1.0, abs(th[k]))
        if dp.names[k].endswith("CGMY.Y"):
            h = 1e-3  # stay on one side of nothing: the series is analytic across poles
        fd = _richardson(lambda t: dp.char_exponent(U, t), th, k, h)
        np.testing.assert_allclose(g[..., k], fd, rtol=1e-8, atol=1e-9, err_msg=dp.names[k])


@pytest.mark.parametrize("proc", FAMS, ids=_ids(FAMS))
def test_hessian_matches_difference_of_exact_gradient(proc: Any) -> None:
    dp = DiffProcess(proc)
    th = dp.values
    _, g, H = dp.char_exponent_hessian(U)
    np.testing.assert_allclose(H, np.swapaxes(H, -1, -2), rtol=1e-13, atol=1e-14)
    _, g1 = dp.char_exponent_and_grad(U)
    np.testing.assert_allclose(g, g1, rtol=1e-14, atol=1e-15)
    for k in range(len(th)):
        h = 1e-3 * max(1.0, abs(th[k]))
        fd = _richardson(lambda t: dp.char_exponent_and_grad(U, t)[1], th, k, h)
        np.testing.assert_allclose(H[..., k], fd, rtol=1e-7, atol=1e-8, err_msg=dp.names[k])


@pytest.mark.parametrize("proc", FAMS, ids=_ids(FAMS))
def test_cumulant_jacobian_matches_richardson(proc: Any) -> None:
    dp = DiffProcess(proc)
    th = dp.values
    k_val, J = dp.cumulants_and_jacobian(4)
    np.testing.assert_allclose(k_val, dp.cumulants(4), rtol=1e-14, atol=1e-16)
    for k in range(len(th)):
        h = 1e-3 * max(1.0, abs(th[k]))
        fd = _richardson(lambda t: dp.cumulants(4, t), th, k, h)
        np.testing.assert_allclose(J[:, k], fd, rtol=1e-8, atol=1e-10, err_msg=dp.names[k])


# ── arbitrary-precision references ──────────────────────────────────────────


def _mp_grad(f: Callable[..., Any], params: List[float], u: float) -> np.ndarray:
    out = []
    for k in range(len(params)):

        def fk(x: Any, k: int = k) -> Any:
            p = [mp.mpf(v) for v in params]
            p[k] = x
            return f(mp.mpf(u), *p)

        out.append(complex(mp.diff(fk, mp.mpf(params[k]))))
    return np.array(out)


def _mp_vg(u: Any, s: Any, n: Any, t: Any) -> Any:
    return -mp.log(1 - 1j * u * t * n + s**2 * n * u**2 / 2) / n


def _mp_nig(u: Any, a: Any, b: Any, d: Any, m: Any) -> Any:
    return 1j * m * u + d * (mp.sqrt(a**2 - b**2) - mp.sqrt(a**2 - (b + 1j * u) ** 2))


def _mp_meixner(u: Any, a: Any, b: Any, d: Any, m: Any) -> Any:
    return 2 * d * (mp.log(mp.cos(b / 2)) - mp.log(mp.cosh((a * u - 1j * b) / 2))) + 1j * m * u


def _mp_cgmy(u: Any, C: Any, G: Any, M: Any, Y: Any) -> Any:
    return C * mp.gamma(-Y) * ((M - 1j * u) ** Y - M**Y + (G + 1j * u) ** Y - G**Y)


def _mp_gh(u: Any, lam: Any, a: Any, b: Any, d: Any, m: Any) -> Any:
    z = mp.sqrt(a**2 - (b + 1j * u) ** 2)
    g = mp.sqrt(a**2 - b**2)
    return (
        1j * m * u + lam * mp.log(g / z) + mp.log(mp.besselk(lam, d * z) / mp.besselk(lam, d * g))
    )


@pytest.mark.parametrize(
    "proc, ref",
    [
        (VarianceGamma(0.2, 0.3, -0.1), _mp_vg),
        (NIG(3.0, -0.5, 0.7, 0.05), _mp_nig),
        (MeixnerProcess(0.4, -0.6, 1.2, 0.02), _mp_meixner),
        (CGMY(1.0, 5.0, 8.0, 0.7), _mp_cgmy),
        (CGMY(1.0, 5.0, 8.0, 0.02), _mp_cgmy),
        (CGMY(1.0, 5.0, 8.0, 1.04), _mp_cgmy),
        (GeneralizedHyperbolicProcess(1.2, 3.0, -0.5, 0.8, 0.03), _mp_gh),
    ],
    ids=["vg", "nig", "meixner", "cgmy", "cgmy-near-0", "cgmy-near-1", "gh"],
)
def test_gradient_against_mpmath(proc: Any, ref: Callable[..., Any]) -> None:
    mp.mp.dps = 40
    dp = DiffProcess(proc)
    for u in (-3.0, 0.4, 2.2):
        psi, g = dp.char_exponent_and_grad(np.array(u))
        exact = _mp_grad(ref, list(dp.values), u)
        val = complex(ref(mp.mpf(u), *[mp.mpf(v) for v in dp.values]))
        assert abs(complex(psi) - val) <= 1e-13 * max(1.0, abs(val))
        np.testing.assert_allclose(g, exact, rtol=1e-11, atol=1e-13)


def test_gh_cumulants_against_mpmath_taylor() -> None:
    mp.mp.dps = 40
    p = [1.2, 3.0, -0.5, 0.8, 0.03]
    dp = DiffProcess(GeneralizedHyperbolicProcess(*p))
    mpp = [mp.mpf(v) for v in p]
    co = mp.taylor(lambda z: _mp_gh(-1j * z, *mpp), 0, 6)
    ref = [float(mp.re(mp.factorial(n) * co[n])) for n in range(1, 7)]
    np.testing.assert_allclose(dp.cumulants(6), ref, rtol=1e-11, atol=1e-14)


def test_cgmy_continuous_across_the_poles() -> None:
    mp.mp.dps = 80
    for Y in (0.0, 1e-6, 0.049, 0.051, 0.95, 1.0, 1.0 + 1e-6, 1.049):
        dp = DiffProcess(CGMY(1.0, 5.0, 8.0, Y))
        y = dp.values[-1]  # the zoo may snap Y onto a pole

        def f(yy: Any) -> Any:
            # removable singularity at the poles: symmetric average, error O(1e-50)
            if yy in (0, 1):
                d = mp.mpf("1e-25")
                return (_mp_cgmy(mp.mpf(2.0), 1, 5, 8, yy + d) + _mp_cgmy(2, 1, 5, 8, yy - d)) / 2
            return _mp_cgmy(mp.mpf(2.0), 1, 5, 8, yy)

        val = f(mp.mpf(y))
        dval = mp.diff(f, mp.mpf(y), h=mp.mpf("1e-20"))
        psi, g = dp.char_exponent_and_grad(np.array(2.0))
        assert abs(complex(psi) - complex(val)) < 1e-13 * abs(complex(val))
        assert abs(g[-1] - complex(dval)) < 1e-11 * abs(complex(dval)), Y


# ── the algebra ─────────────────────────────────────────────────────────────


def test_sum_rule_concatenates_gradients() -> None:
    vg, dr = VarianceGamma(0.2, 0.3, -0.1), Drift(0.01)
    s = DiffProcess(vg + dr)
    a, b = DiffProcess(vg), DiffProcess(dr)
    psi, g = s.char_exponent_and_grad(U)
    pa, ga = a.char_exponent_and_grad(U)
    pb, gb = b.char_exponent_and_grad(U)
    np.testing.assert_allclose(psi, pa + pb, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(g, np.concatenate([ga, gb], axis=-1), rtol=1e-15, atol=1e-16)
    np.testing.assert_allclose(s.cumulants(4), a.cumulants(4) + b.cumulants(4), rtol=1e-14)


def test_scale_rule_and_its_derivative() -> None:
    c = 0.7
    base = NIG(3.0, -0.5, 0.7, 0.05)
    dp = DiffProcess(c * base)
    db = DiffProcess(base)
    psi, g = dp.char_exponent_and_grad(U)
    pb, gb = db.char_exponent_and_grad(c * U)
    np.testing.assert_allclose(psi, pb, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(g[..., :4], gb, rtol=1e-14, atol=1e-15)
    # d/dc psi(c u) = u psi'(u c): compare with the exact u-derivative
    names = dp.names
    assert names[-1] == "scale"
    _, gu = DiffProcess(base).char_exponent_and_grad(c * U)
    h = 1e-4
    dpsi_du = (db.char_exponent(c * U + h) - db.char_exponent(c * U - h)) / (2 * h)
    np.testing.assert_allclose(g[..., -1], U * dpsi_du, rtol=1e-7, atol=1e-9)
    kb = db.cumulants(4)
    np.testing.assert_allclose(dp.cumulants(4), kb * c ** np.arange(1, 5), rtol=1e-14)
    _, J = dp.cumulants_and_jacobian(4)
    n = np.arange(1, 5)
    np.testing.assert_allclose(J[:, -1], n * c ** (n - 1) * kb, rtol=1e-13)


def test_subordination_reproduces_variance_gamma_exactly() -> None:
    # VG(sigma, nu, theta) = BM(theta, sigma) @ Gamma(1/nu, 1/nu)
    s, nu, th = 0.2, 0.3, -0.1
    sub = DiffProcess(BrownianMotion(th, s) @ GammaProcess(1 / nu, 1 / nu))
    vg = DiffProcess(VarianceGamma(s, nu, th))
    np.testing.assert_allclose(sub.char_exponent(U), vg.char_exponent(U), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(sub.cumulants(5), vg.cumulants(5), rtol=1e-13)
    _, g_sub = sub.char_exponent_and_grad(U)
    _, g_vg = vg.char_exponent_and_grad(U)
    # chain rule: theta and sigma are shared; d/dnu = -(1/nu^2)(d/da + d/db)
    np.testing.assert_allclose(g_sub[..., 0], g_vg[..., 2], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(g_sub[..., 1], g_vg[..., 0], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(
        -(g_sub[..., 2] + g_sub[..., 3]) / nu**2, g_vg[..., 1], rtol=1e-12, atol=1e-14
    )


def test_nested_algebra_against_richardson() -> None:
    proc = 0.5 * (BrownianMotion(0.0, 0.25) @ InverseGaussianProcess(1.0, 2.0)) + Drift(0.03)
    dp = DiffProcess(proc)
    th = dp.values
    _, g = dp.char_exponent_and_grad(U)
    for k in range(len(th)):
        fd = _richardson(lambda t: dp.char_exponent(U, t), th, k, 1e-3 * max(1, abs(th[k])))
        np.testing.assert_allclose(g[..., k], fd, rtol=1e-8, atol=1e-10)
    c = proc.cumulants(4)
    np.testing.assert_allclose(dp.cumulants(4), [c[n] for n in range(1, 5)], rtol=1e-10)


# ── API behaviour ─────────────────────────────────────────────────────────────


def test_free_selection_names_and_round_trip() -> None:
    proc = VarianceGamma(0.2, 0.3, -0.1) + Drift(0.01)
    dp = DiffProcess(proc, free=["sigma", Param(0.0, "Drift.mu")])
    assert dp.names == ("VarianceGamma.sigma", "Drift.mu")
    np.testing.assert_allclose(dp.values, [0.2, 0.01])
    _, g = dp.char_exponent_and_grad(U)
    assert g.shape == U.shape + (2,)
    new = dp.to_process([0.25, 0.02])
    assert isinstance(new, type(proc))
    assert proc._left.sigma == 0.2  # type: ignore[attr-defined]
    np.testing.assert_allclose(
        DiffProcess(new).char_exponent(U), dp.char_exponent(U, [0.25, 0.02]), rtol=1e-15
    )
    with pytest.raises(KeyError, match="ambiguous"):
        DiffProcess(BrownianMotion(0, 1) + Drift(0.1), free=["mu"])
    with pytest.raises(KeyError, match="no parameter"):
        DiffProcess(proc, free=["kappa"])
    with pytest.raises(ValueError):
        dp.char_exponent(U, [1.0])
    with pytest.raises(ValueError, match="twice"):
        DiffProcess(proc, free=["sigma", "VarianceGamma.sigma"])


def test_repeated_family_gets_indexed_names() -> None:
    dp = DiffProcess(GammaProcess(1.0, 2.0) + GammaProcess(3.0, 4.0))
    assert dp.names == (
        "GammaProcess.a",
        "GammaProcess.b",
        "GammaProcess[1].a",
        "GammaProcess[1].b",
    )


def test_alpha_stable_parameter_gradient_and_no_cumulants() -> None:
    dp = DiffProcess(AlphaStable(1.6, 0.3, 0.4, 0.01))
    assert not dp.analytic
    u = np.array([-2.0, 0.0, 0.5, 3.0])
    psi, g = dp.char_exponent_and_grad(u)
    assert np.all(np.isfinite(g))
    th = dp.values
    for k in range(4):
        fd = _richardson(lambda t: dp.char_exponent(u, t), th, k, 1e-4)
        np.testing.assert_allclose(g[..., k], fd, rtol=1e-8, atol=1e-10)
    with pytest.raises(UnsupportedProcessError):
        dp.cumulants(2)
    with pytest.raises(ValueError, match="real u"):
        dp.char_exponent(np.array([1j]))


def test_unsupported_process_is_refused() -> None:
    from spxa.zoo.levy.compound_poisson import CompoundPoisson, NormalJump

    with pytest.raises(UnsupportedProcessError, match="no differentiable closed form"):
        DiffProcess(CompoundPoisson(rate=1.0, jump_dist=NormalJump(0.0, 0.1)))


def test_story_and_repr() -> None:
    dp = DiffProcess(VarianceGamma(0.2, 0.3, -0.1) + Drift(0.01))
    s = dp.__story__()
    assert "EXACT" in s and "VarianceGamma(sigma=0.2" in s
    assert "DiffProcess(" in repr(dp)


def test_char_func_and_log_mgf_consistency() -> None:
    dp = DiffProcess(NIG(3.0, -0.5, 0.7, 0.05))
    t = 0.37
    np.testing.assert_allclose(dp.char_func(U, t), np.exp(t * dp.char_exponent(U)), rtol=1e-15)
    z = np.array([-1.5, 0.2, 2.0])  # inside alpha +- beta
    km = dp.log_mgf(z)
    np.testing.assert_allclose(np.imag(km), 0, atol=1e-15)
    kappa = dp.cumulants(2)
    h = 1e-4
    np.testing.assert_allclose(
        (np.real(dp.log_mgf(h)) - np.real(dp.log_mgf(-h))) / (2 * h), kappa[0], rtol=1e-7
    )
