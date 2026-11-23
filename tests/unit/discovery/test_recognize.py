"""recognize(): zoo processes and their compositions as discovery models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spxa.discovery.recognize import NotRecognizedError, recognize
from spxa.zoo.levy import (
    CGMY,
    NIG,
    AlphaStable,
    BrownianMotion,
    CompoundPoisson,
    DoubleExponentialJump,
    Drift,
    GammaProcess,
    InverseGaussianProcess,
    KouJumpDiffusion,
    MeixnerProcess,
    MertonJumpDiffusion,
    NormalJump,
    PoissonProcess,
    UniformJump,
    VarianceGamma,
)

PRIMITIVES = {
    "drift": lambda: Drift(0.2),
    "bm": lambda: BrownianMotion(0.05, 0.3),
    "gamma": lambda: GammaProcess(2.0, 4.0),
    "ig": lambda: InverseGaussianProcess(0.8, 1.7),
    "vg": lambda: VarianceGamma(0.2, 0.35, -0.15),
    "nig": lambda: NIG(6.0, -2.0, 0.3, 0.05),
    "meixner": lambda: MeixnerProcess(0.4, -0.7, 1.3, 0.05),
    "poisson": lambda: PoissonProcess(1.5, 0.2),
    "poisson_down": lambda: PoissonProcess(1.5, -0.2),
    "cpn": lambda: CompoundPoisson(0.9, NormalJump(-0.1, 0.25)),
    "cpde": lambda: CompoundPoisson(2.0, DoubleExponentialJump(0.35, 12.0, 7.0)),
    "cgmy": lambda: CGMY(0.7, 5.0, 8.0, 0.6),
    "merton": lambda: MertonJumpDiffusion(0.05, 0.2, 1.0, -0.05, 0.15),
    "kou": lambda: KouJumpDiffusion(0.05, 0.2, 1.0, 0.4, 10.0, 5.0),
}
U = np.array([-2.0, 0.5, 3.0])


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_primitives_keep_their_law(name: str) -> None:
    proc = PRIMITIVES[name]()
    model = recognize(proc)
    zoo = proc.cumulants(4)
    np.testing.assert_allclose(
        model.cumulants(4), [float(zoo[n]) for n in range(1, 5)], rtol=1e-9, atol=1e-13
    )
    cf = np.array([complex(proc.char_func(float(u), t=0.7)) for u in U])
    np.testing.assert_allclose(model.char_func(U, t=0.7), cf, rtol=1e-9, atol=1e-13)


def test_compositions() -> None:
    vg, g, bm = VarianceGamma(0.2, 0.35, -0.15), GammaProcess(2.0, 4.0), BrownianMotion(0.0, 0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expr = vg + 0.5 * g - Drift(0.1)
    m = recognize(expr)
    assert m.structure.key() == "VG + 0.5*Gamma + -Drift"
    expected = VarianceGamma(0.2, 0.35, -0.15).cumulants(4)
    k = m.cumulants(4)
    assert k[0] == pytest.approx(-0.15 + 0.5 * 0.5 - 0.1)
    assert k[1] == pytest.approx(float(expected[2]) + 0.25 * 0.125)
    sub = recognize(bm @ g)
    assert sub.structure.key() == "(BM) @ (Gamma)"
    # BM(σ) in Γ(a, b) time: variance σ² a/b, kurtosis 3/a in excess
    k = sub.cumulants(4)
    assert k[1] == pytest.approx(0.09 * 0.5)
    assert k[3] / k[1] ** 2 == pytest.approx(3 / 2.0)


def test_unrecognized() -> None:
    with pytest.raises(NotRecognizedError):
        recognize(AlphaStable(alpha=1.5, beta=0.0))
    with pytest.raises(NotRecognizedError):
        recognize(CompoundPoisson(1.0, UniformJump(-1, 1)))
    with pytest.raises(NotRecognizedError):
        recognize(CGMY(1.0, 5.0, 5.0, 1.5))
    with pytest.raises(NotRecognizedError):
        recognize(KouJumpDiffusion(0.0, 0.2, 1.0, 1.0, 10.0, 5.0))
    with pytest.raises(NotRecognizedError):
        recognize(PoissonProcess(1.0, 0.0))
