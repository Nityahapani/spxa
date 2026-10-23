"""Analytic finite-horizon count moments for a Markov-modulated Poisson process."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm

from ._mmpp_engine import _validated_parameters


def count_moments(
    Q: ArrayLike,  # noqa: N803
    rates: ArrayLike,
    horizon: float,
    *,
    initial_state: int = 0,
    initial_probabilities: ArrayLike | None = None,
) -> tuple[float, float]:
    """Return exact-in-model mean and variance via deterministic matrix quadrature.

    This is a numerical evaluation of the finite-horizon moment equations,
    not an exact-arithmetic result. Requires SciPy.
    """
    off, exits, lam, t_end, p0 = _validated_parameters(
        Q, rates, horizon, initial_state, initial_probabilities, 1, 1
    )
    n = lam.size
    p = np.eye(n)[initial_state] if p0 is None else p0
    generator = off.copy()
    np.fill_diagonal(generator, -exits)
    if t_end == 0:
        return 0.0, 0.0
    # Augmented ODE for E[N(t) 1_{J(t)=j}] and second factorial moment.
    # Row vectors: p'=pQ, m'=mQ+p diag(lambda), f'=fQ+2m diag(lambda).
    aug = np.zeros((3 * n, 3 * n), dtype=float)
    for k in range(3):
        aug[k * n : (k + 1) * n, k * n : (k + 1) * n] = generator.T
    diag_rates = np.diag(lam)
    aug[n : 2 * n, 0:n] = diag_rates
    aug[2 * n : 3 * n, n : 2 * n] = 2 * diag_rates
    initial = np.zeros(3 * n)
    initial[:n] = p
    out = expm(aug * t_end) @ initial
    mean = float(np.sum(out[n : 2 * n]))
    factorial2 = float(np.sum(out[2 * n : 3 * n]))
    variance = factorial2 + mean - mean**2
    return mean, max(0.0, variance) if variance > -1e-8 else variance


def stationary_distribution(Q: ArrayLike) -> np.ndarray:  # noqa: N803
    """Compute stationary distribution only when it is unique.

    An irreducible chain always has a unique stationary distribution, but
    reducible chains with one closed class may also have a unique one.
    """
    q = np.asarray(Q, dtype=float)
    n = q.shape[0]
    _validated_parameters(q, np.zeros(n), 0.0, 0, None, 1, 1)
    a = q.T.copy()
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    if np.linalg.matrix_rank(a) != n:
        raise ValueError("stationary distribution is not unique")
    pi = np.linalg.solve(a, b)
    if np.min(pi) < -1e-10:
        raise ArithmeticError("stationary solution has negative components")
    pi = np.maximum(pi, 0)
    return np.asarray(pi / pi.sum())


def _stirling2_table(order: int) -> list[list[int]]:
    """Stirling numbers of the second kind S(n, k) for 0 <= k <= n <= order."""
    s = [[0] * (order + 1) for _ in range(order + 1)]
    s[0][0] = 1
    for i in range(1, order + 1):
        for k in range(1, i + 1):
            s[i][k] = k * s[i - 1][k] + s[i - 1][k - 1]
    return s


def count_cumulants(
    Q: ArrayLike,  # noqa: N803
    rates: ArrayLike,
    horizon: float,
    order: int,
    *,
    initial_state: int = 0,
    initial_probabilities: ArrayLike | None = None,
) -> dict[int, float]:
    """Return ``{1: kappa_1, ..., order: kappa_order}`` for the count N(horizon).

    Factorial moments solve ``u_k' = u_k Q + k u_{k-1} diag(rates)``; one block
    matrix exponential gives all of them. They are converted to cumulants through
    the factorial-moment generating series (a logarithm) and Stirling numbers.
    Cancellation costs roughly ``(order - 1) * log10(1 + max(rates) * horizon)``
    digits of the 16 available, so requests that would lose more than 9 digits
    raise ``ValueError`` instead of returning noise. Use a shorter horizon or a
    lower order.
    """
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    order = int(order)
    off, exits, lam, t_end, p0 = _validated_parameters(
        Q, rates, horizon, initial_state, initial_probabilities, 1, 1
    )
    n = lam.size
    if t_end == 0:
        return {k: 0.0 for k in range(1, order + 1)}
    digits_lost = (order - 1) * math.log10(1.0 + float(lam.max()) * t_end)
    if digits_lost > 9.0:
        raise ValueError(
            f"cumulants up to order {order} are ill-conditioned for max(rates)*horizon = "
            f"{float(lam.max()) * t_end:.3g} (about {digits_lost:.0f} of 16 digits would be lost); "
            "use a shorter horizon or a lower order"
        )
    p = np.eye(n)[initial_state] if p0 is None else p0
    generator = off.copy()
    np.fill_diagonal(generator, -exits)
    big = np.zeros(((order + 1) * n, (order + 1) * n), dtype=float)
    d = np.diag(lam)
    for k in range(order + 1):
        big[k * n : (k + 1) * n, k * n : (k + 1) * n] = generator.T
        if k >= 1:
            big[k * n : (k + 1) * n, (k - 1) * n : k * n] = k * d
    start = np.zeros((order + 1) * n)
    start[:n] = p
    out = expm(big * t_end) @ start
    # factorial moments F_k = E[N(N-1)...(N-k+1)], F_0 = 1
    fm = [float(np.sum(out[k * n : (k + 1) * n])) for k in range(order + 1)]
    # series G(s) = sum F_k s^k / k!; factorial cumulants are k! [s^k] log G(s)
    g = [fm[k] / math.factorial(k) for k in range(order + 1)]
    h = [0.0] * (order + 1)  # h = log G, from G' = G h'
    for k in range(1, order + 1):
        acc = k * g[k]
        for j in range(1, k):
            acc -= j * h[j] * g[k - j]
        h[k] = acc / (k * g[0])
    fcum = [h[k] * math.factorial(k) for k in range(order + 1)]
    s2 = _stirling2_table(order)
    return {m: float(sum(s2[m][k] * fcum[k] for k in range(1, m + 1))) for m in range(1, order + 1)}
