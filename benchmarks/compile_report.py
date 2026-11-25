"""
Speed of spxa.compile against the uncompiled paths (prints a Markdown table).

Run: ``python benchmarks/compile_report.py``. Times are the best of several
repeats on the current machine; ratios are what matter. Baselines are the
fastest uncompiled route: a leaf's closed-form ``char_exponent``, a composite's
``_char_exponent``, :class:`~spxa.graph.autodiff.process.DiffProcess` for
gradients, and its COS pricer once per parameter set for batches.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Tuple

import numpy as np

from spxa.graph.autodiff.fourier import price_and_gradient
from spxa.graph.autodiff.process import DiffProcess
from spxa.zoo.levy import (
    CGMY,
    NIG,
    BrownianMotion,
    GammaProcess,
    InverseGaussianProcess,
    KouJumpDiffusion,
    MertonJumpDiffusion,
    VarianceGamma,
)


def best(fn: Callable[[], object], repeat: int = 7) -> float:
    fn()
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out) * 1e3


def main() -> None:
    u = np.linspace(-50.0, 50.0, 4096)
    strikes = np.linspace(80.0, 120.0, 9)
    merton = MertonJumpDiffusion(0.05, 0.2, 1.0, -0.05, 0.1)
    comp = (
        BrownianMotion(0.05, 0.2) + 0.5 * KouJumpDiffusion(0.05, 0.2, 2.0, 0.4, 8.0, 5.0)
    ) @ GammaProcess(4.0, 4.0) + CGMY(1.0, 5.0, 6.0, 0.5)
    vec = (
        (NIG(10.0, -2.0, 0.5) + VarianceGamma(0.2, 0.3, -0.1)) @ GammaProcess(5.0, 5.0)
        + 0.5 * NIG(8.0, 1.0, 0.3)
    ) @ InverseGaussianProcess(1.0, 2.0)
    dp_comp, dp_vec = DiffProcess(comp), DiffProcess(vec)
    base = {
        "merton": best(lambda: merton.char_exponent(u)),
        "comp": best(lambda: comp._char_exponent(u), 3),
        "vec": best(lambda: vec._char_exponent(u)),
        "gcomp": best(lambda: dp_comp.char_exponent_and_grad(u)),
        "gvec": best(lambda: dp_vec.char_exponent_and_grad(u)),
        "cos1": best(lambda: price_and_gradient(dp_comp, 100.0, strikes, 0.5), 5),
    }
    base["cos"] = 64 * base["cos1"]
    rows: List[Tuple[str, float, float]] = []
    for backend in ("numpy", "numba", "jax"):
        try:
            cm, cc, cv = merton.compile(backend), comp.compile(backend), vec.compile(backend)
        except Exception as exc:  # noqa: BLE001 - optional backends
            print(f"{backend}: unavailable ({exc})")
            continue
        theta = np.tile(cc.values, (64, 1))

        def cos(cc: Any = cc, theta: np.ndarray = theta) -> object:
            return cc.cos_price(100.0, strikes, 0.5, theta=theta, grad=True)

        rows += [
            (
                f"Merton psi, 4096 points [{backend}]",
                base["merton"],
                best(lambda: cm.char_exponent(u)),
            ),
            (
                f"composite psi with Kou, 4096 points [{backend}]",
                base["comp"],
                best(lambda: cc.char_exponent(u)),
            ),
            (
                f"composite psi, vectorised families [{backend}]",
                base["vec"],
                best(lambda: cv.char_exponent(u)),
            ),
            (
                f"composite psi + grad ({cc.n_params} params) [{backend}]",
                base["gcomp"],
                best(lambda: cc.char_exponent_and_grad(u)),
            ),
            (
                f"vectorised psi + grad ({cv.n_params} params) [{backend}]",
                base["gvec"],
                best(lambda: cv.char_exponent_and_grad(u)),
            ),
            (
                f"COS price + Jacobian, 9 strikes [{backend}]",
                base["cos1"],
                best(lambda cc=cc: cc.cos_price(100.0, strikes, 0.5, grad=True), 5),
            ),
            (
                f"COS price + Jacobian, 9 strikes x 64 parameter sets [{backend}]",
                base["cos"],
                best(cos, 3),
            ),
        ]
    print("| quantity | uncompiled (ms) | compiled (ms) | speed-up |")
    print("|---|---:|---:|---:|")
    for name, a, b in rows:
        print(f"| {name} | {a:.2f} | {b:.2f} | {a / b:.1f}x |")


if __name__ == "__main__":
    main()
