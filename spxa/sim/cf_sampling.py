r"""
Sampling from a characteristic function by FFT inversion.

Some infinitely divisible laws (CGMY, Meixner, tempered stable) have a
closed-form characteristic function but no simple sampler. This module
recovers the density on a grid with one FFT and inverts the CDF.

Grid
    The window is :math:`[m - L, m + L]`, with mean :math:`m`. The half-width
    :math:`L` covers ``n_sd`` standard deviations, and ``tail_decay``
    e-foldings of any exponential tail the caller names. With
    :math:`N` points, :math:`\Delta x = 2L/N` and :math:`\Delta u = 2\pi/(N\Delta x)`,
    so the frequency grid reaches :math:`u_{\max} = \pi/\Delta x`. :math:`N`
    doubles until :math:`|\varphi(u_{\max})|` falls below ``cf_tol``.
    Otherwise the density is under-resolved.

Density
    With :math:`x_k = x_0 + k\Delta x` and :math:`u_j = (j - N/2)\Delta u`,

    .. math::
        f(x_k) \approx \frac{\Delta u}{2\pi}\sum_j \varphi(u_j)e^{-iu_jx_k}
        = \frac{\Delta u}{2\pi}(-1)^k\,\mathrm{FFT}\big[\varphi(u_j)e^{-iu_jx_0}\big]_k,

    which is exact up to aliasing (mass outside the window) and truncation
    (frequencies beyond :math:`u_{\max}`). The CDF integrates this density
    with the trapezoidal rule at the grid points, so interpolated quantiles
    carry no half-cell shift. Monotonicity is enforced only afterwards,
    because ripples near a kink cancel in the integral.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["CharacteristicFunctionNotDecayed", "cf_grid_cdf", "sample_from_cf"]


class CharacteristicFunctionNotDecayed(ValueError):
    """The cf stays large at the finest grid: the law has atoms or lives on a lattice."""


def cf_grid_cdf(
    char_func: Callable[[np.ndarray], np.ndarray],
    mean: float,
    std: float,
    *,
    n_sd: float = 14.0,
    tail_decay: Optional[float] = None,
    lower: Optional[float] = None,
    n_points: int = 2**14,
    max_points: int = 2**20,
    cf_tol: float = 1e-10,
    fail_above: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``(x, cdf)`` on a grid fine enough that the cf has decayed.

    ``tail_decay`` is the slowest exponential rate of the tails (for CGMY,
    ``min(G, M)``). When it is given, the window also covers ``36 / tail_decay``,
    the distance over which an exponential tail drops by :math:`e^{-36}`.
    ``lower`` is a known lower end of the support (0 for subordinators). The
    window then starts there, so no draw falls below it. If ``fail_above``
    is set and :math:`|\varphi(u_{\max})|` still exceeds it at ``max_points``,
    :class:`CharacteristicFunctionNotDecayed` is raised. Laws with atoms
    (a compound-Poisson clock) or on a lattice (integer-valued parents)
    have no density to invert.
    """
    if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
        raise ValueError(f"need a finite mean and a positive std, got {mean}, {std}")
    half = n_sd * std
    if tail_decay is not None and tail_decay > 0:
        half = max(half, 36.0 / tail_decay)
    n = n_points
    while True:
        dx = 2 * half / n
        du = 2 * np.pi / (n * dx)
        u = (np.arange(n) - n / 2) * du
        edge = abs(complex(np.atleast_1d(char_func(u[:1]))[0]))
        if edge <= cf_tol:
            break
        if n >= max_points:
            if fail_above is not None and edge > fail_above:
                raise CharacteristicFunctionNotDecayed(
                    f"|phi(u_max)| = {edge:.3g} at {n} points: no density to invert"
                )
            break
        n *= 2
    x0 = mean - half if lower is None else max(mean - half, lower)
    phi = np.atleast_1d(np.asarray(char_func(u), dtype=complex))
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    dens = (du / (2 * np.pi)) * sign * np.fft.fft(phi * np.exp(-1j * u * x0))
    pdf = dens.real
    x = x0 + np.arange(n) * dx
    # integrate before removing ripples: oscillations around a kink or a
    # singularity cancel in the integral, but clipping them first biases the CDF
    raw = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dx)])
    cdf = np.maximum.accumulate(np.clip(raw / raw[-1], 0.0, 1.0))
    return x, cdf


def sample_from_cf(
    char_func: Callable[[np.ndarray], np.ndarray],
    mean: float,
    std: float,
    size: int | tuple[int, ...],
    rng: np.random.Generator,
    **grid: float,
) -> np.ndarray:
    """Draw ``size`` variates whose characteristic function is ``char_func``."""
    x, cdf = cf_grid_cdf(char_func, mean, std, **grid)  # type: ignore[arg-type]
    # strictly increasing abscissae for interp: drop flat stretches of the CDF
    keep = np.concatenate([[True], np.diff(cdf) > 0])
    return np.interp(rng.uniform(size=size), cdf[keep], x[keep])
