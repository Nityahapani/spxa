"""Exact Hull-White price gradients (``spxa.rates.hw_sensitivities``)."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np
import pytest

from spxa.rates.capfloor import cap_floor
from spxa.rates.hull_white import HullWhite
from spxa.rates.hw_calibration import calibrate_hull_white, model_price
from spxa.rates.hw_sensitivities import hull_white_price_and_gradient
from spxa.rates.indices import EURIBOR6M
from spxa.rates.model_cashflows import swap_cashflows
from spxa.rates.swaptions import swaption
from tests.unit.rates.test_capfloor import PROJ
from tests.unit.rates.test_hull_white import CURVE

REF = "2024-03-15"
ST = [1.0, 3.0, 6.0]
MODELS = {
    "flat": HullWhite(CURVE, 0.06, 0.0105),
    "piecewise": HullWhite(CURVE, 0.02, [0.011, 0.009, 0.0085, 0.007], ST),
    "fast": HullWhite(CURVE, 0.3, [0.02, 0.006], [2.5]),
}
SWAPTIONS = [
    swaption(REF, e, t, k, payer=p, notional=1e4)
    for e, t in (("1Y", "2Y"), ("5Y", "10Y"), ("3Y", "4Y"))
    for k in (0.02, 0.034, 0.05)
    for p in (True, False)
]
CAPS = [
    cap_floor(REF, "5Y", 0.036, EURIBOR6M, notional=1e4),
    cap_floor(REF, "10Y", 0.03, EURIBOR6M, notional=1e4, kind="floor"),
]


def _theta(m: HullWhite) -> np.ndarray:
    return np.r_[m.sigma, m.a]


def _rebuild(m: HullWhite, th: np.ndarray) -> HullWhite:
    return HullWhite(m.curve, float(th[-1]), th[:-1], m.sigma_times)


def _richardson(f: Callable[[np.ndarray], float], th: np.ndarray, rel: float = 1e-3) -> np.ndarray:
    out = np.zeros_like(th)
    for k in range(th.size):
        h = rel * th[k]

        def d(hh: float) -> float:
            up, dn = th.copy(), th.copy()
            up[k] += hh
            dn[k] -= hh
            return (f(up) - f(dn)) / (2 * hh)

        out[k] = (4 * d(h / 2) - d(h)) / 3
    return out


# -- mpmath reference: an independent 30-digit implementation of the HW formulas ----------


def _mp_y(mp: Any, m: HullWhite, sig: list, a: Any, t: float) -> Any:
    st = list(m.sigma_times)
    edges = [0.0] + [u for u in st if u < t] + [t]
    tot = mp.mpf(0)
    for u0, u1 in zip(edges[:-1], edges[1:]):
        j = sum(1 for u in st if u <= u0)
        tot += sig[j] ** 2 * (mp.exp(-2 * a * (t - u1)) - mp.exp(-2 * a * (t - u0))) / (2 * a)
    return tot


def _mp_B(mp: Any, a: Any, dt: float) -> Any:
    return -mp.expm1(-a * mp.mpf(dt)) / a


def _mp_swaption(mp: Any, m: HullWhite, sw: Any, th: list) -> Any:
    """``P(0,T) * int V+(x) phi(x) dx`` with the root of ``V`` found by ``findroot``."""
    sig, a = th[:-1], th[-1]
    T = float(m.curve.time(sw.expiry))
    cf = swap_cashflows(sw.swap, m.curve, PROJ)
    live = cf.times > T + 1e-12
    times, amounts = cf.times[live], cf.amounts[live]
    assert float(cf.amounts[~live].sum()) == 0.0
    pT = mp.mpf(float(m.curve.discount(T)))
    ratio = [mp.mpf(float(m.curve.discount(t))) / pT for t in times]
    yT = _mp_y(mp, m, sig, a, T)
    Bs = [_mp_B(mp, a, t - T) for t in times]
    amts = [mp.mpf(float(x)) for x in amounts]

    def V(x: Any) -> Any:
        return mp.fsum(c * r * mp.exp(-b * x - b * b * yT / 2) for c, r, b in zip(amts, ratio, Bs))

    sd = mp.sqrt(yT)
    root = mp.findroot(V, (-12 * sd, 12 * sd), solver="anderson")
    dens = lambda x: mp.exp(-x * x / (2 * yT)) / (sd * mp.sqrt(2 * mp.pi))  # noqa: E731
    side = (-12 * sd, root) if V(root - sd) > 0 else (root, 12 * sd)
    return pT * mp.quad(lambda x: V(x) * dens(x), [side[0], side[1]])


def _mp_caplet(mp: Any, m: HullWhite, cpl: Any, th: list) -> Any:
    sig, a = th[:-1], th[-1]
    c, crv = cpl.coupon, m.curve
    tf = float(crv.time(c.fixing))
    s, e, p = (float(crv.time(d)) for d in (c.start, c.end, c.payment_date))
    Q0 = float(crv.discount(s) / crv.discount(e))
    Ks = 1.0 + c.accrual * (cpl.strike - (cpl.forward(PROJ) - (Q0 - 1.0) / c.accrual))
    assert Ks > 0 and tf > 0
    v = abs(_mp_B(mp, a, e - tf) - _mp_B(mp, a, s - tf)) * mp.sqrt(_mp_y(mp, m, sig, a, tf))
    d1 = (mp.log(mp.mpf(Q0) / Ks) + v * v / 2) / v
    d2 = d1 - v
    N = mp.ncdf
    und = Q0 * N(d1) - Ks * N(d2) if cpl.kind == "cap" else Ks * N(-d2) - Q0 * N(-d1)
    return float(c.notional * crv.discount(p)) * und


def _mp_gradient(mp: Any, price: Callable[[list], Any], th: np.ndarray) -> np.ndarray:
    base = [mp.mpf(float(x)) for x in th]
    out = []
    for k in range(th.size):

        def f(x: Any, k: int = k) -> Any:
            t = list(base)
            t[k] = x
            return price(t)

        out.append(float(mp.diff(f, base[k])))
    return np.array(out)


@pytest.mark.parametrize("name", ["flat", "piecewise", "fast"])
@pytest.mark.parametrize("idx", [0, 3, 7, 8, 13, 16])
def test_swaption_gradient_matches_30_digit_reference(name: str, idx: int) -> None:
    mp = pytest.importorskip("mpmath")
    m, sw = MODELS[name], SWAPTIONS[idx]
    th = _theta(m)
    with mp.workdps(30):
        ref_p = float(_mp_swaption(mp, m, sw, [mp.mpf(float(x)) for x in th]))
        ref_g = _mp_gradient(mp, lambda t: _mp_swaption(mp, m, sw, t), th)
    for method in ("jamshidian", "integral"):
        p, g = hull_white_price_and_gradient(m, sw, PROJ, method=method)
        assert p == pytest.approx(ref_p, rel=1e-12)
        assert np.max(np.abs(g - ref_g)) <= 1e-11 * np.max(np.abs(ref_g))


@pytest.mark.parametrize("name", ["flat", "piecewise", "fast"])
def test_caplet_gradient_matches_30_digit_reference(name: str) -> None:
    mp = pytest.importorskip("mpmath")
    m = MODELS[name]
    th = _theta(m)
    for cap in CAPS:
        for cpl in (cap.caplets[0], cap.caplets[len(cap.caplets) // 2], cap.caplets[-1]):
            with mp.workdps(30):
                ref_g = _mp_gradient(mp, lambda t: _mp_caplet(mp, m, cpl, t), th)
            p, g = hull_white_price_and_gradient(m, cpl, PROJ)
            assert p == pytest.approx(model_price(m, cpl, PROJ), rel=1e-13)
            assert np.max(np.abs(g - ref_g)) <= 1e-12 * np.max(np.abs(ref_g))


@pytest.mark.parametrize("name", ["flat", "piecewise", "fast"])
def test_price_and_gradient_agree_with_the_pricer_on_every_instrument(name: str) -> None:
    m = MODELS[name]
    th = _theta(m)
    for inst in SWAPTIONS[::2] + CAPS:
        p, g = hull_white_price_and_gradient(m, inst, PROJ)
        assert p == pytest.approx(model_price(m, inst, PROJ), rel=1e-12)
        fd = _richardson(lambda t, inst=inst: model_price(_rebuild(m, t), inst, PROJ), th)
        assert np.max(np.abs(g - fd)) <= 1e-6 * np.max(np.abs(fd))


def test_pieces_after_the_last_fixing_have_zero_gradient() -> None:
    m = HullWhite(CURVE, 0.05, [0.01, 0.008, 0.007], [2.0, 4.0])
    _, g = hull_white_price_and_gradient(m, SWAPTIONS[0], PROJ)  # 1Y expiry
    assert g[1] == 0.0 and g[2] == 0.0 and g[0] > 0.0


def test_jamshidian_and_integral_branches_agree() -> None:
    m = MODELS["piecewise"]
    for sw in SWAPTIONS:
        pj, gj = hull_white_price_and_gradient(m, sw, PROJ, method="jamshidian")
        pi, gi = hull_white_price_and_gradient(m, sw, PROJ, method="integral")
        assert pi == pytest.approx(pj, rel=1e-12)
        assert np.max(np.abs(gi - gj)) <= 1e-11 * np.max(np.abs(gj))


def test_expired_and_intrinsic_instruments_have_zero_gradient() -> None:
    m = MODELS["flat"]
    fwd = swaption(REF, "1Y", "5Y", 0.03, notional=1e4)
    sw = dataclasses.replace(fwd, expiry=CURVE.reference_date)  # expiry T = 0: intrinsic
    assert float(CURVE.time(sw.expiry)) == 0.0
    p, g = hull_white_price_and_gradient(m, sw, PROJ)
    assert p == model_price(m, sw, PROJ)
    assert np.all(g == 0.0)
    # caplet already fixed (skip_first=False keeps the spot period)
    fixed = cap_floor(REF, "2Y", 0.03, EURIBOR6M, notional=1e4, skip_first=False).caplets[0]
    assert float(CURVE.time(fixed.coupon.fixing)) == 0.0
    p, g = hull_white_price_and_gradient(m, fixed, PROJ)
    assert p == pytest.approx(model_price(m, fixed, PROJ), rel=1e-13, abs=1e-12)
    assert np.all(g == 0.0)
    # caplet with a negative effective strike K* <= 0: intrinsic, so constant
    cpl = cap_floor(REF, "2Y", -150.0, EURIBOR6M, notional=1e4).caplets[0]
    p, g = hull_white_price_and_gradient(m, cpl, PROJ)
    assert p > 0.0
    assert p == pytest.approx(model_price(m, cpl, PROJ), rel=1e-13)
    assert np.all(g == 0.0)


def test_rejects_unknown_inputs() -> None:
    m = MODELS["flat"]
    with pytest.raises(ValueError):
        hull_white_price_and_gradient(m, SWAPTIONS[0], PROJ, method="nope")
    with pytest.raises(TypeError):
        hull_white_price_and_gradient(m, object(), PROJ)


# -- calibration ------------------------------------------------------------------------


def test_global_calibration_with_exact_jacobian_recovers_piecewise_truth() -> None:
    st = [2.0, 5.0]
    truth = HullWhite(CURVE, 0.05, [0.012, 0.009, 0.0075], st)
    insts = [
        swaption(REF, e, t, 0.034, notional=1e4) for e in ("1Y", "3Y", "6Y") for t in ("2Y", "5Y")
    ]
    tgt = [model_price(truth, i, PROJ) for i in insts]
    for jac in ("exact", "2-point"):
        cal = calibrate_hull_white(
            CURVE, insts, tgt, PROJ, a=None, mode="global", sigma_times=st, jac=jac
        )
        assert cal.model.a == pytest.approx(0.05, rel=1e-9)
        assert cal.model.sigma == pytest.approx(truth.sigma, rel=1e-9)
        assert cal.max_relative_error < 1e-12


def test_calibration_rejects_unknown_jac() -> None:
    with pytest.raises(ValueError):
        calibrate_hull_white(CURVE, CAPS, [1.0, 1.0], PROJ, mode="global", jac="3-point")
