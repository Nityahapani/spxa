"""
CGMY process (Carr–Geman–Madan–Yor).

A four-parameter generalisation of the VG model that nests VG (Y=0),
compound Poisson (Y < 0), and α-stable (C+=C-, G=M=0) as special cases.
Controls tail heaviness, jump activity, and asymmetry independently.

Lévy density (Carr et al. 2002, equation 1):
  k(x) = C · exp(-M·x)/x^{1+Y} 𝟙_{x>0} + C · exp(-G·|x|)/|x|^{1+Y} 𝟙_{x<0}

Parameter constraints:
  C > 0, G ≥ 0, M ≥ 0, Y < 2.
  For Y ∈ [1, 2): infinite variation; for Y ∈ (0, 1): finite variation, infinite activity;
  for Y ≤ 0: finite activity.

References
----------
Carr, P., Geman, H., Madan, D.B. & Yor, M. (2002). The fine structure of
asset returns: an empirical investigation. *Journal of Business*, 75(2),
305–332.

Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.
Chapter 4, Table 4.2.
"""

from __future__ import annotations

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import DensityLevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


def _cgmy_levy_measure(C: float, G: float, M: float, Y: float) -> DensityLevyMeasure:
    """
    CGMY Lévy density.

    k(x) = C·exp(-M·x)/x^{1+Y}·𝟙_{x>0} + C·exp(-G·|x|)/|x|^{1+Y}·𝟙_{x<0}

    Carr et al. (2002), equation (1).
    """
    from scipy.special import gamma as gamma_fn

    def density(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        pos = x > 0
        neg = x < 0
        out[pos] = C * np.exp(-M * x[pos]) / x[pos] ** (1 + Y)
        out[neg] = C * np.exp(-G * np.abs(x[neg])) / np.abs(x[neg]) ** (1 + Y)
        return out

    def total_mass() -> float:
        if Y >= 0:
            return np.inf
        from scipy import integrate

        pos_mass, _ = integrate.quad(lambda x: C * np.exp(-G * x) / x ** (1 + Y), 0, np.inf)
        neg_mass, _ = integrate.quad(lambda x: C * np.exp(-M * x) / x ** (1 + Y), 0, np.inf)
        return float(pos_mass + neg_mass)

    def tail(xv: float) -> float:
        from scipy import integrate

        result, _ = integrate.quad(
            lambda t: C * np.exp(-M * t) / t ** (1 + Y), xv, np.inf, limit=200
        )
        return float(result)

    def moment_1() -> float:
        return float(C * gamma_fn(1 - Y) * (M ** (Y - 1) - G ** (Y - 1)))

    def moment_2() -> float:
        return float(C * gamma_fn(2 - Y) * (M ** (Y - 2) + G ** (Y - 2)))

    def moment_3() -> float:
        return float(C * gamma_fn(3 - Y) * (M ** (Y - 3) - G ** (Y - 3)))

    def moment_4() -> float:
        return float(C * gamma_fn(4 - Y) * (M ** (Y - 4) + G ** (Y - 4)))

    return DensityLevyMeasure(
        _density_fn=density,
        _total_mass=total_mass(),
        _tail_fn=tail,
        _moment_fns={1: moment_1, 2: moment_2, 3: moment_3, 4: moment_4},
        _tail_moment_test=lambda n: (G > 0 and M > 0) or n < Y,
        _minimum_moment_order=2 if Y >= 1 else 1,
    )


class CGMY(Process):
    """
    CGMY (Carr–Geman–Madan–Yor) process.

    A flexible pure-jump Lévy model with independent control over:
    - Overall jump intensity (C)
    - Left tail decay (G)
    - Right tail decay (M)
    - Fine structure / activity (Y)

    Parameters
    ----------
    C :
        Overall jump intensity. C > 0.
    G :
        Left-side exponential decay rate. G ≥ 0.
        G=0 gives a stable left tail.
    M :
        Right-side exponential decay rate. M ≥ 0.
        M=0 gives a stable right tail.
    Y :
        Fine-structure index. Y < 2.
        Y < 0  → finite activity (compound Poisson)
        Y ∈ [0,1) → infinite activity, finite variation
        Y ∈ [1,2) → infinite activity, infinite variation
        Y → 2 approximates stable process

    Notes
    -----
    Special cases:
    - Y = 0: Variance Gamma (with appropriate C, G, M)
    - G = M, Y = 0: symmetric VG
    - G = M → 0, Y ∈ (0,2): α-stable

    Cumulants (Carr et al. 2002, equation 2):
      κ_n = C · Γ(n-Y) · (M^{Y-n} + (-1)^n · G^{Y-n})

    References
    ----------
    Carr, P., Geman, H., Madan, D.B. & Yor, M. (2002). The fine structure of
    asset returns. *Journal of Business*, 75(2), 305–332.
    """

    def __init__(self, C: float, G: float, M: float, Y: float) -> None:
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if G < 0:
            raise ValueError(f"G must be non-negative, got {G}")
        if M < 0:
            raise ValueError(f"M must be non-negative, got {M}")
        if not all(np.isfinite(x) for x in (C, G, M, Y)):
            raise ValueError("Parameters must be finite")
        if Y <= 0 and (G == 0 or M == 0):
            raise ValueError("Y <= 0 requires positive G and M for an integrable tail")
        if Y >= 2:
            raise ValueError(f"Y must be < 2, got {Y}")

        self.C = C
        self.G = G
        self.M = M
        self.Y = Y

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CGMY(C={C}, G={G}, M={M}, Y={Y})",
            math_note=(
                "Positive tail: C*exp(-M*x)/x^(1+Y); "
                "negative tail: C*exp(-G*|x|)/|x|^(1+Y); "
                f"activity: {'finite' if Y < 0 else 'infinite'}, "
                f"variation: {'finite' if Y < 1 else 'infinite'}"
            ),
            reference="Carr, Geman, Madan & Yor (2002), eq. (1); Cont & Tankov (2004) Table 4.2",
        )
        super().__init__(exactness=ExactnessLevel.EXACT, _node=node)

    def _triplet(self) -> LevyTriplet:
        from scipy.integrate import quad

        nu = _cgmy_levy_measure(self.C, self.G, self.M, self.Y)

        def paired(x: float) -> float:
            return float(
                self.C * np.exp(-self.G * x) * np.expm1(-(self.M - self.G) * x) / x**self.Y
            )

        drift = quad(paired, 0, 1, epsabs=1e-10)[0]
        return LevyTriplet(float(drift), 0.0, nu)

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=True,
            is_martingale=False,
            has_finite_variance=(self.G > 0 and self.M > 0),
            has_finite_mean=(self.Y > 1 or (self.G > 0 and self.M > 0)),
            self_similarity_index=None,
            tail_index=self.Y if self.G == 0 or self.M == 0 else None,
            is_subordinator=False,
        )

    def cumulant_exact(self, n: int, t: float = 1.0) -> float:
        """CGMY cumulants, including the Y=0 and Y=1 limits.

        All moments exist for strictly tempered tails, even at Y>=1.
        The first cumulant is an analytic-continuation expression, not
        the divergent uncompensated small-jump first moment.

        References
        ----------
        Carr, Geman, Madan & Yor (2002), characteristic exponent.
        Sato (1999), Theorem 25.3 (tail moment criterion).
        """
        from scipy.special import gamma

        if n < 1 or t < 0:
            raise ValueError("Positive cumulant order and nonnegative time required")
        if (self.G == 0 or self.M == 0) and n >= self.Y:
            raise ValueError("Requested tail moment does not exist without tempering")
        if n == 1 and self.Y == 1:
            return float(t * self.C * np.log(self.G / self.M))
        if n == 1 and self.G > 0 and self.M > 0:
            p = self.Y - 1
            return float(
                -t
                * self.C
                * gamma(2 - self.Y)
                * self.G**p
                * np.expm1(p * np.log(self.M / self.G))
                / p
            )
        return float(
            t
            * self.C
            * gamma(n - self.Y)
            * (self.M ** (self.Y - n) + (-1) ** n * self.G ** (self.Y - n))
        )

    def cumulants(self, order: int) -> dict[int, float]:
        return {n: self.cumulant_exact(n) for n in range(1, order + 1)}

    def char_func_exact(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        """Evaluate the continuous CGMY exponent, with removable poles handled analytically."""
        return np.exp(t * self.characteristic_exponent(u))

    def char_func(self, u: complex | float | np.ndarray, t: float = 1.0) -> np.ndarray:
        return self.char_func_exact(u=u, t=t)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulation via characteristic function inversion (Gil-Pelaez).

        For each time step, samples from the CGMY distribution by inverting
        the characteristic function numerically. This is a numerical approximation
        but slower than closed-form methods.

        For production use with many paths, a series representation method
        (Rosiński 2001) should be preferred. This implementation is provided
        for correctness; the series method will be added in a future release.

        Parameters
        ----------
        n_steps, n_paths, T, rng :
            Standard simulation parameters.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1).
        """
        from scipy import integrate

        rng = rng or np.random.default_rng()
        dt = T / n_steps

        def cgmy_pdf(x: float, t: float) -> float:
            def integrand(u: float) -> float:
                cf = complex(self.char_func_exact(u=u, t=t))
                return float((cf * np.exp(-1j * u * x)).real)

            result, _ = integrate.quad(integrand, 0, 500, limit=300)
            return float(result / np.pi)

        u_grid = np.linspace(-300, 300, 2000)
        du = u_grid[1] - u_grid[0]
        cf_vals = self.char_func_exact(u=u_grid, t=dt)

        N = len(u_grid)
        x_grid = np.fft.fftfreq(N, d=du / (2 * np.pi))
        x_grid = np.fft.fftshift(x_grid)

        pdf_fft = (
            np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(cf_vals))).real * du * N / (2 * np.pi)
        )
        pdf_fft = np.maximum(pdf_fft, 0)
        cdf = np.cumsum(pdf_fft) * (x_grid[1] - x_grid[0])
        cdf /= cdf[-1]

        uniforms = rng.uniform(size=(n_paths, n_steps))
        increments = np.interp(uniforms, cdf, x_grid)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths

    def __repr__(self) -> str:
        return f"CGMY(C={self.C}, G={self.G}, M={self.M}, Y={self.Y})"

    def characteristic_exponent(self, u: complex | float | np.ndarray) -> np.ndarray:
        """Carr et al. (2002) exponent; M tempers positive, G negative jumps."""
        from scipy.special import gamma

        z = np.asarray(u, dtype=complex)
        if (
            np.any(z.imag < -self.M)
            or np.any(z.imag > self.G)
            or (self.Y <= 0 and (np.any(z.imag == -self.M) or np.any(z.imag == self.G)))
        ):
            raise ValueError("Exponential moment outside the CGMY convergence strip")
        if self.Y == 0:
            return np.asarray(-self.C * (np.log1p(-1j * z / self.M) + np.log1p(1j * z / self.G)))
        if self.Y == 1:

            def f(w: complex | float | np.ndarray) -> np.ndarray:
                w = np.asarray(w, dtype=complex)
                return np.where(w == 0, 0, w * np.log(np.where(w == 0, 1, w)))

            return np.asarray(
                self.C * (f(self.M - 1j * z) - f(self.M) + f(self.G + 1j * z) - f(self.G))
            )
        if min(abs(self.Y), abs(self.Y - 1)) < 1e-5:
            import mpmath as mp

            with mp.workdps(45):
                y = mp.mpf(self.Y)
                m = mp.mpf(self.M)
                g = mp.mpf(self.G)
                values = [
                    complex(
                        self.C
                        * mp.gamma(-y)
                        * ((m - 1j * complex(v)) ** y - m**y + (g + 1j * complex(v)) ** y - g**y)
                    )
                    for v in z.flat
                ]
            return np.asarray(values).reshape(z.shape)
        return np.asarray(
            self.C
            * gamma(-self.Y)
            * (
                (self.M - 1j * z) ** self.Y
                - self.M**self.Y
                + (self.G + 1j * z) ** self.Y
                - self.G**self.Y
            )
        )
