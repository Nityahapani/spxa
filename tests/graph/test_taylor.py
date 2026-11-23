"""Truncated Taylor series (cumulants by exact series arithmetic)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
import scipy.special as sps

from spxa.graph.autodiff.jet import variables
from spxa.graph.autodiff.taylor import TaylorError, TSeries, taylor_variable

mp = pytest.importorskip("mpmath")

N = 9


@pytest.fixture(autouse=True)
def _restore_mp_precision() -> Any:
    dps = mp.mp.dps
    yield
    mp.mp.dps = dps


CASES: list[tuple[str, Callable[[Any], Any], Callable[[Any], Any], complex]] = [
    ("exp", np.exp, mp.exp, 0.3),
    ("log", np.log, mp.log, 2.0),
    ("sqrt", np.sqrt, mp.sqrt, 3.0),
    ("pow-real", lambda z: z**0.3, lambda z: z**0.3, 1.5),
    ("pow-int", lambda z: z**5, lambda z: z**5, -0.7),
    ("pow-neg", lambda z: z**-2.5, lambda z: z**-2.5, 0.9),
    ("rpow", lambda z: 2.0**z, lambda z: mp.mpf(2) ** z, 0.4),
    ("seriespow", lambda z: z**z, lambda z: z**z, 1.3),
    ("div", lambda z: 1 / (2 - z * z), lambda z: 1 / (2 - z * z), 0.0),
    ("cosh", np.cosh, mp.cosh, 0.3),
    ("sinh", np.sinh, mp.sinh, -0.2),
    ("cos", np.cos, mp.cos, 0.7),
    ("sin", np.sin, mp.sin, 0.7),
    ("expm1", np.expm1, mp.expm1, 1e-9),
    ("log1p", np.log1p, mp.log1p, 1e-9),
    ("square", np.square, lambda z: z * z, 1.7),
    ("reciprocal", np.reciprocal, lambda z: 1 / z, 1.7),
    ("kve", lambda z: sps.kve(0.7, z), lambda z: mp.besselk(0.7, z) * mp.exp(z), 1.2),
    ("kv-complex", lambda z: sps.kv(2.3, z), lambda z: mp.besselk(2.3, z), 0.8 + 0.3j),
    ("kv-int", lambda z: sps.kv(1.0, z), lambda z: mp.besselk(1, z), 0.5),
    (
        "complex-log",
        lambda z: np.log(1 - 0.4j * z + 0.5 * z * z),
        lambda z: mp.log(1 - 0.4j * z + 0.5 * z * z),
        0.0,
    ),
    (
        "meixner-logcosh",
        lambda z: np.log(np.cosh((0.4 * (-1j * z) + 0.6j) / 2)),
        lambda z: mp.log(mp.cosh((0.4 * (-1j * z) + 0.6j) / 2)),
        0.0,
    ),
]


@pytest.mark.parametrize("name, f, fm, z0", CASES, ids=[c[0] for c in CASES])
def test_series_against_mpmath(name: str, f: Any, fm: Any, z0: complex) -> None:
    mp.mp.dps = 40
    s = f(taylor_variable(z0, N))
    assert isinstance(s, TSeries) and s.order == N
    ref = mp.taylor(fm, mp.mpmathify(z0), N)
    for k in range(N + 1):
        assert abs(complex(s.coeffs[k]) - complex(ref[k])) <= 1e-13 * max(
            1.0, abs(complex(ref[k]))
        ), (
            name,
            k,
        )


def test_reversed_operands_and_scalars() -> None:
    z = taylor_variable(0.5, 4)
    for s, ref in [
        (3.0 - z, [2.5, -1, 0, 0, 0]),
        (2.0 * z + 1.0, [2.0, 2, 0, 0, 0]),
        (np.add(1.0, z), [1.5, 1, 0, 0, 0]),
        (np.subtract(1.0, z), [0.5, -1, 0, 0, 0]),
        (np.multiply(2.0, z), [1.0, 2, 0, 0, 0]),
        (-z, [-0.5, -1, 0, 0, 0]),
    ]:
        np.testing.assert_allclose([complex(c) for c in s.coeffs], ref, atol=1e-15)
    q = np.true_divide(1.0, z)  # 1/(0.5 + h) = 2 - 4h + 8h^2 - ...
    np.testing.assert_allclose([complex(c) for c in q.coeffs], [2, -4, 8, -16, 32], rtol=1e-15)
    assert q.derivative(2) == pytest.approx(16.0)


def test_jet_coefficients_carry_exact_parameter_gradients() -> None:
    # log(1 - z/a) = -sum z^k / (k a^k): d/da of the k-th coefficient is 1/a^{k+1}
    (a,) = variables([0.7], order=2)
    s = np.log(1 - taylor_variable(0.0, 6) / a)
    for k in range(1, 7):
        c = s.coeffs[k]
        assert complex(c.value) == pytest.approx(-1 / (k * 0.7**k), rel=1e-14)
        assert complex(c.grad[0]) == pytest.approx(1 / 0.7 ** (k + 1), rel=1e-14)
        assert complex(c.hess[0, 0]) == pytest.approx(-(k + 1) / 0.7 ** (k + 2), rel=1e-13)


def test_bessel_with_jet_order_matches_difference() -> None:
    (nu,) = variables([0.7])
    s = sps.kve(nu, taylor_variable(1.2, 4))
    h = 1e-6
    for k in range(5):
        hi = sps.kve(0.7 + h, taylor_variable(1.2, 4)).coeffs[k]
        lo = sps.kve(0.7 - h, taylor_variable(1.2, 4)).coeffs[k]
        assert complex(s.coeffs[k].grad[0]) == pytest.approx(
            complex((hi - lo) / (2 * h)), rel=1e-7, abs=1e-10
        )


def test_refusals() -> None:
    z = taylor_variable(1.0, 3)
    with pytest.raises(TaylorError):
        float(z)
    with pytest.raises(TaylorError):
        bool(z)
    with pytest.raises(TaylorError):
        np.asarray(z)
    with pytest.raises(TaylorError):
        np.abs(z)
    with pytest.raises(TaylorError):
        np.sum(z)  # type: ignore[call-overload]
    with pytest.raises(TaylorError):
        sps.kve(z, 1.0)
    with pytest.raises(TaylorError):
        taylor_variable(0.0, 0)
