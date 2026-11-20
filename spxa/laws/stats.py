r"""
Statistical expectations for laws about random output.

A law about a simulator cannot demand equality. It can demand that a test
with a known false-alarm rate does not reject. Every test here returns a
p-value that is exactly or asymptotically uniform when the null holds, and
:func:`expect_not_rejected` fails the law when the p-value falls below
``level``.

False alarms
    With per-example level :math:`\alpha`, a check of :math:`n` examples of a
    correct law fails with probability at most :math:`n\alpha` (Bonferroni).
    :data:`LEVEL` is :math:`10^{-6}`, so 100 examples give a false alarm at
    most once in :math:`10^4` runs. Seeds come from the law's strategies, so
    a run is reproducible, and a reported counterexample replays exactly.

Tests
    :func:`ecf_test`
        Empirical characteristic function at frequency :math:`u`. For
        :math:`n` draws, :math:`\operatorname{Re}` and :math:`\operatorname{Im}`
        of :math:`\bar\varphi_n(u) = n^{-1}\sum_k e^{iuX_k}` have variances
        :math:`[(1 \pm \operatorname{Re}\varphi(2u))/2 - (\operatorname{Re}/\operatorname{Im}\varphi(u))^2]/n`,
        because :math:`\cos^2 = (1 + \cos 2\cdot)/2`. The summands are
        bounded, so the CLT applies for every law, including heavy-tailed
        ones. Several frequencies combine by Bonferroni.
    :func:`mean_test`
        A z-test of :math:`\bar X` against a known mean and variance.
    :func:`variance_test`
        A z-test of the sample variance :math:`s^2`, using
        :math:`\operatorname{Var}s^2 = \kappa_4/n + 2\kappa_2^2/(n-1)`.
    :func:`independence_test`
        The joint empirical cf of a pair :math:`(A, B)` against
        :math:`\varphi_A(u)\varphi_B(v)`, which is what independence of
        increments means.

Degenerate laws (zero variance, such as a pure drift) have no sampling
noise. There, each test demands agreement to ``1e-9`` and returns a p-value
of 1 or 0.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats as _st  # type: ignore[import-untyped]

from spxa.laws.expect import LawViolation

__all__ = [
    "LEVEL",
    "ecf_test",
    "expect_not_rejected",
    "independence_test",
    "mean_test",
    "variance_test",
]

LEVEL = 1e-6
_DEGENERATE = 1e-14


def _two_sided(z: float) -> float:
    return float(2.0 * _st.norm.sf(abs(z)))


def _z_or_exact(diff: float, var: float) -> float:
    """Two-sided p-value of ``diff`` with variance ``var``, exact when the variance vanishes."""
    if var <= _DEGENERATE:
        return 1.0 if abs(diff) <= 1e-9 else 0.0
    return _two_sided(diff / np.sqrt(var))


def _bonferroni(pvalues: Sequence[float]) -> float:
    return float(min(1.0, len(pvalues) * min(pvalues)))


def ecf_test(
    samples: np.ndarray,
    char_func: Callable[[float], complex],
    frequencies: Sequence[float],
) -> float:
    """p-value of the empirical cf against ``char_func`` at the given frequencies."""
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    pvalues = []
    for u in frequencies:
        p, p2 = complex(char_func(u)), complex(char_func(2.0 * u))
        m = complex(np.mean(np.exp(1j * u * x)))
        var_re = max((1.0 + p2.real) / 2.0 - p.real**2, 0.0) / n
        var_im = max((1.0 - p2.real) / 2.0 - p.imag**2, 0.0) / n
        pvalues += [_z_or_exact(m.real - p.real, var_re), _z_or_exact(m.imag - p.imag, var_im)]
    return _bonferroni(pvalues)


def mean_test(samples: np.ndarray, mean: float, variance: float) -> float:
    """p-value of the sample mean against ``mean``, given the true ``variance``."""
    x = np.asarray(samples, dtype=float).ravel()
    return _z_or_exact(float(np.mean(x)) - mean, variance / x.size)


def variance_test(samples: np.ndarray, kappa2: float, kappa4: float) -> float:
    """p-value of the sample variance against ``kappa2``, given the fourth cumulant."""
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    s2 = float(np.var(x, ddof=1))
    return _z_or_exact(s2 - kappa2, kappa4 / n + 2.0 * kappa2**2 / (n - 1))


def independence_test(
    a: np.ndarray,
    b: np.ndarray,
    cf_a: Callable[[float], complex],
    cf_b: Callable[[float], complex],
    pairs: Sequence[tuple[float, float]],
) -> float:
    r"""p-value of :math:`E e^{i(uA + vB)} = \varphi_A(u)\varphi_B(v)` at the given ``(u, v)``."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = a.size
    pvalues = []
    for u, v in pairs:
        p = complex(cf_a(u)) * complex(cf_b(v))
        p2 = complex(cf_a(2.0 * u)) * complex(cf_b(2.0 * v))
        m = complex(np.mean(np.exp(1j * (u * a + v * b))))
        var_re = max((1.0 + p2.real) / 2.0 - p.real**2, 0.0) / n
        var_im = max((1.0 - p2.real) / 2.0 - p.imag**2, 0.0) / n
        pvalues += [_z_or_exact(m.real - p.real, var_re), _z_or_exact(m.imag - p.imag, var_im)]
    return _bonferroni(pvalues)


def expect_not_rejected(pvalue: float, level: float = LEVEL, what: str = "") -> None:
    """Fail the law when ``pvalue < level``."""
    if not pvalue >= level:
        raise LawViolation(
            f"{what or 'statistical test'} rejected: p = {pvalue:.3g} < {level:g}",
            {"p_value": pvalue, "level": level},
        )
