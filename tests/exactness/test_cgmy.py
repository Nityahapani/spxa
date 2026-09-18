"""
Exactness tests for CGMY process.

Each test verifies a closed-form result against a published formula.
These tests must never be skipped or marked xfail.
"""

import numpy as np
import pytest

from spxa.zoo.levy.cgmy import CGMY


@pytest.mark.exactness
def test_cgmy_invalid_params_raise() -> None:
    """
    CGMY requires C>0, G≥0, M≥0, Y<2.
    Carr et al. (2002), equation (1) parameter constraints.
    """
    with pytest.raises(ValueError):
        CGMY(C=0.0, G=1.0, M=1.0, Y=0.5)
    with pytest.raises(ValueError):
        CGMY(C=1.0, G=-1.0, M=1.0, Y=0.5)
    with pytest.raises(ValueError):
        CGMY(C=1.0, G=1.0, M=1.0, Y=2.0)


@pytest.mark.exactness
def test_cgmy_no_gaussian_component() -> None:
    """
    CGMY is a pure-jump process: σ²=0 in its Lévy triplet.
    Carr et al. (2002): CGMY has no Brownian component.
    """
    cgmy = CGMY(C=1.0, G=5.0, M=5.0, Y=0.5)
    assert cgmy.triplet.sigma_sq == pytest.approx(0.0)


@pytest.mark.exactness
def test_cgmy_char_func_at_zero() -> None:
    """
    φ(0; t) = 1. Carr et al. (2002), equation (5).
    """
    cgmy = CGMY(C=1.0, G=5.0, M=5.0, Y=0.5)
    cf = cgmy.char_func_exact(u=0.0, t=1.0)
    assert abs(complex(cf) - 1.0) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.exactness
def test_cgmy_char_func_modulus_leq_1() -> None:
    """
    |φ(u)| ≤ 1 for all u. Standard property of characteristic functions.
    """
    cgmy = CGMY(C=1.0, G=5.0, M=5.0, Y=0.5)
    u_vals = np.linspace(-5.0, 5.0, 50)
    cf_vals = cgmy.char_func_exact(u=u_vals, t=1.0)
    assert np.all(np.abs(cf_vals) <= 1.0 + 1e-10)


@pytest.mark.exactness
def test_cgmy_char_func_time_additivity() -> None:
    """
    φ(u; s+t) = φ(u; s)·φ(u; t). Lévy process property.
    Sato (1999), Definition 1.6.
    """
    cgmy = CGMY(C=1.0, G=5.0, M=5.0, Y=0.5)
    u, s, t = 1.0, 0.4, 0.6
    cf_s = complex(cgmy.char_func_exact(u=u, t=s))
    cf_t = complex(cgmy.char_func_exact(u=u, t=t))
    cf_st = complex(cgmy.char_func_exact(u=u, t=s + t))
    assert abs(cf_s * cf_t - cf_st) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.exactness
def test_cgmy_cumulant_1() -> None:
    """
    κ_1 = C·Γ(1-Y)·(M^{Y-1} - G^{Y-1}).
    Carr et al. (2002), equation (2).
    """
    from scipy.special import gamma as gamma_fn
    C, G, M, Y = 1.0, 5.0, 7.0, 0.5
    cgmy = CGMY(C=C, G=G, M=M, Y=Y)
    expected = C * float(gamma_fn(1 - Y)) * (M ** (Y - 1) - G ** (Y - 1))
    assert cgmy.cumulant_exact(1) == pytest.approx(expected, rel=1e-10)


@pytest.mark.exactness
def test_cgmy_cumulant_2() -> None:
    """
    κ_2 = C·Γ(2-Y)·(M^{Y-2} + G^{Y-2}).
    Carr et al. (2002), equation (2).
    """
    from scipy.special import gamma as gamma_fn
    C, G, M, Y = 1.0, 5.0, 7.0, 0.5
    cgmy = CGMY(C=C, G=G, M=M, Y=Y)
    expected = C * float(gamma_fn(2 - Y)) * (M ** (Y - 2) + G ** (Y - 2))
    assert cgmy.cumulant_exact(2) == pytest.approx(expected, rel=1e-10)


@pytest.mark.exactness
def test_cgmy_symmetric_when_g_equals_m() -> None:
    """
    CGMY with G=M is symmetric: characteristic function is real for real u.
    φ(u) is real iff the distribution is symmetric about 0. Carr et al. (2002).
    """
    cgmy = CGMY(C=1.0, G=5.0, M=5.0, Y=0.5)
    u_vals = np.array([0.5, 1.0, 2.0])
    cf_vals = cgmy.char_func_exact(u=u_vals, t=1.0)
    assert np.all(np.abs(cf_vals.imag) < 1e-10)


@pytest.mark.exactness
def test_cgmy_finite_variance_when_y_lt_2() -> None:
    """
    CGMY has finite variance iff Y < 2.
    Carr et al. (2002): κ_2 exists iff 2 > Y.
    """
    assert CGMY(C=1.0, G=5.0, M=5.0, Y=0.5).properties.has_finite_variance is True
    assert CGMY(C=1.0, G=5.0, M=5.0, Y=1.5).properties.has_finite_variance is True


@pytest.mark.exactness
def test_cgmy_cumulant_order_below_y_raises() -> None:
    """
    κ_n does not exist for n ≤ Y. cumulant_exact must raise ValueError.
    Carr et al. (2002): Γ(n-Y) diverges when n ≤ Y.
    """
    cgmy = CGMY(C=1.0, G=5.0, M=5.0, Y=1.5)
    with pytest.raises(ValueError, match="does not exist"):
        cgmy.cumulant_exact(1)
