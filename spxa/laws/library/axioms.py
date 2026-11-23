r"""
Axioms of Lévy processes, stated on their characteristic functions and
cumulants.

For a Lévy process :math:`X` with :math:`\varphi_t(u) = E[e^{iuX_t}]`:

- :math:`\varphi_t(0) = 1` (``cf_at_zero``);
- :math:`|\varphi_t(u)| \le 1` (``cf_is_bounded``);
- :math:`\varphi_t(-u) = \overline{\varphi_t(u)}`, since :math:`X_t` is real
  (``cf_is_hermitian``);
- :math:`\varphi_{s+t} = \varphi_s\varphi_t`, from stationary independent
  increments (``cf_semigroup``);
- the even cumulants are non-negative. :math:`\kappa_2 = \sigma^2 + \int
  x^2\nu(dx)` and :math:`\kappa_{2k} = \int x^{2k}\nu(dx)` for
  :math:`k \ge 2` (Sato 1999, Thm 25.3) (``even_cumulants_nonnegative``).

The process argument defaults to light-tailed compositions of zoo
primitives (:func:`~spxa.laws.processes.compositions` without
subordination). Pass other strategies to :class:`~spxa.laws.law.Law` to
test other processes.
"""

from __future__ import annotations

import numpy as np

from spxa.core.process import Process
from spxa.laws.expect import expect, expect_close
from spxa.laws.law import Law, law
from spxa.laws.library._common import ATOL, RTOL, frequencies, times
from spxa.laws.processes import compositions
from spxa.laws.quantities import char_func, cumulants

_PROCESSES = compositions(max_depth=1, operations=("+", "*", "-"))


@law(X=_PROCESSES, t=times(), tags=("axiom", "cf"))
def cf_at_zero(X: Process, t: float) -> None:
    """phi_t(0) = 1"""
    expect_close(char_func(X, 0.0, t), 1.0, rtol=RTOL, atol=ATOL, what="phi_t(0)")


@law(X=_PROCESSES, u=frequencies(), t=times(), tags=("axiom", "cf"))
def cf_is_bounded(X: Process, u: float, t: float) -> None:
    """|phi_t(u)| <= 1"""
    modulus = float(np.abs(char_func(X, u, t))[0])
    expect(modulus <= 1.0 + RTOL, f"|phi_t(u)| = {modulus!r} exceeds 1", modulus=modulus)


@law(X=_PROCESSES, u=frequencies(), t=times(), tags=("axiom", "cf"))
def cf_is_hermitian(X: Process, u: float, t: float) -> None:
    """phi_t(-u) = conj(phi_t(u))"""
    expect_close(
        char_func(X, -u, t),
        np.conj(char_func(X, u, t)),
        rtol=RTOL,
        atol=ATOL,
        what="phi_t(-u) vs conj phi_t(u)",
    )


@law(X=_PROCESSES, u=frequencies(), s=times(), t=times(), tags=("axiom", "cf"))
def cf_semigroup(X: Process, u: float, s: float, t: float) -> None:
    """phi_{s+t}(u) = phi_s(u) phi_t(u)"""
    expect_close(
        char_func(X, u, s + t),
        char_func(X, u, s) * char_func(X, u, t),
        rtol=RTOL,
        atol=ATOL,
        what="phi_{s+t} vs phi_s phi_t",
    )


@law(X=_PROCESSES, tags=("axiom", "cumulants"))
def even_cumulants_nonnegative(X: Process) -> None:
    """kappa_2 >= 0 and kappa_4 >= 0"""
    k = cumulants(X, 4)
    expect(k[1] >= -ATOL, f"kappa_2 = {k[1]!r} < 0", kappa_2=k[1])
    expect(k[3] >= -ATOL, f"kappa_4 = {k[3]!r} < 0", kappa_4=k[3])


AXIOMS: tuple[Law, ...] = (
    cf_at_zero,
    cf_is_bounded,
    cf_is_hermitian,
    cf_semigroup,
    even_cumulants_nonnegative,
)
