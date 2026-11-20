r"""
Discovering risk-neutral structures from option prices.

Given European option quotes, :func:`discover_from_options` searches the
same grammar as :func:`~spxa.discovery.search.discover`. It ranks the
compositions whose risk-neutral prices (:func:`~spxa.discovery.pricing.cos_prices`)
best reproduce the quotes.

Objective
    Each structure is fitted by least squares on prices divided by the
    Black vega at the market implied volatility,

    .. math::
        \min_\theta \sum_i \Big(\frac{P_i(\theta) - P_i^{mkt}}{\mathcal V_i}\Big)^2,

    which to first order is the sum of squared implied-volatility errors
    (Cont & Tankov 2004, §13.2). Vegas are floored at :math:`10^{-4} S_0\sqrt{T}`
    so that far out-of-the-money quotes cannot dominate. After the fit, the
    model's implied volatilities are computed exactly.

Ranking
    Treating the implied-volatility errors as i.i.d. Gaussian gives
    :math:`\ell = -\tfrac n2\log(\mathrm{SSE}/n)` up to a constant. AIC and BIC
    are therefore comparable across structures fitted to the same quotes.
    They are a heuristic, as the errors are neither independent nor Gaussian.

Fit quality
    ``extras["iv_rmse"]`` and ``extras["iv_max_error"]`` hold the
    implied-volatility errors (absolute, as decimals).
    ``extras["price_rel_rmse"]`` holds the RMS of relative price errors.

A Lévy model has a single law, scaled in time, for all maturities. A
surface quoted at several maturities therefore also tests whether the
smile decays as :math:`T^{-1/2}`, which Lévy models do and real markets
often do not. :func:`discover_from_options` reports the RMSE per maturity
so that this misfit can be seen.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np
from scipy import optimize

from spxa.discovery.expr import Model, Node
from spxa.discovery.fit import FitResult, initial_params
from spxa.discovery.grammar import Grammar, canonical
from spxa.discovery.pricing import cos_prices
from spxa.discovery.report import DiscoveryReport, screen_degenerate
from spxa.market.black import black_greeks
from spxa.market.implied import implied_black_volatility

_PENALTY = 1e12


@dataclass(frozen=True)
class OptionQuotes:
    """European option quotes on one underlying.

    Parameters
    ----------
    spot :
        Current price of the underlying.
    strikes, maturities, prices :
        One entry per quote; maturities in years.
    call :
        Booleans, ``True`` for calls; a single bool applies to all quotes.
    rate, dividend :
        Continuously compounded rate and dividend yield.
    """

    spot: float
    strikes: np.ndarray
    maturities: np.ndarray
    prices: np.ndarray
    call: np.ndarray
    rate: float = 0.0
    dividend: float = 0.0

    @classmethod
    def from_arrays(
        cls,
        spot: float,
        strikes: Any,
        maturities: Any,
        prices: Any,
        call: Any = True,
        rate: float = 0.0,
        dividend: float = 0.0,
    ) -> OptionQuotes:
        K = np.asarray(strikes, dtype=float).ravel()
        T = np.broadcast_to(np.asarray(maturities, dtype=float), K.shape).copy()
        P = np.asarray(prices, dtype=float).ravel()
        C = np.broadcast_to(np.asarray(call, dtype=bool), K.shape).copy()
        if not (K.shape == P.shape):
            raise ValueError("strikes and prices must have the same length")
        if np.any(K <= 0) or np.any(T <= 0) or spot <= 0:
            raise ValueError("spot, strikes and maturities must be positive")
        return cls(float(spot), K, T, P, C, float(rate), float(dividend))

    def __len__(self) -> int:
        return int(self.strikes.size)

    @property
    def forwards(self) -> np.ndarray:
        return self.spot * np.exp((self.rate - self.dividend) * self.maturities)

    @property
    def discounts(self) -> np.ndarray:
        return np.exp(-self.rate * self.maturities)

    def implied_vols(self, prices: Optional[np.ndarray] = None) -> np.ndarray:
        """Black implied volatilities of ``prices`` (default: the quotes); NaN outside arbitrage bounds."""
        p = self.prices if prices is None else prices
        return np.asarray(
            implied_black_volatility(
                p, self.forwards, self.strikes, self.maturities, self.discounts, self.call
            ),
            dtype=float,
        )

    def maturity_groups(self) -> list[tuple[float, np.ndarray]]:
        return [
            (float(T), np.flatnonzero(self.maturities == T)) for T in np.unique(self.maturities)
        ]


def model_prices(model: Model, quotes: OptionQuotes, n_terms: int = 1024) -> np.ndarray:
    """COS prices of every quote under ``model``."""
    out = np.empty(len(quotes))
    for T, idx in quotes.maturity_groups():
        out[idx] = cos_prices(
            model,
            quotes.spot,
            quotes.strikes[idx],
            T,
            quotes.rate,
            quotes.dividend,
            quotes.call[idx],
            n_terms=n_terms,
        )
    return out


def _vegas(quotes: OptionQuotes, iv: np.ndarray) -> np.ndarray:
    sigma = np.where(np.isfinite(iv) & (iv > 0), iv, np.nanmedian(iv))
    g = black_greeks(
        quotes.forwards, quotes.strikes, quotes.maturities, sigma, quotes.discounts, quotes.call
    )
    vega = np.asarray(g["vega"], dtype=float)
    return np.maximum(vega, 1e-4 * quotes.spot * np.sqrt(quotes.maturities))


def fit_options(
    structure: Node,
    quotes: OptionQuotes,
    *,
    n_starts: int = 3,
    rng: Optional[np.random.Generator] = None,
    initial: Optional[Sequence[Sequence[float]]] = None,
    n_terms: int = 1024,
    maxiter: int = 300,
) -> FitResult:
    """Fit ``structure`` to option quotes (see the module docstring)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    iv_mkt = quotes.implied_vols()
    if np.all(~np.isfinite(iv_mkt)):
        raise ValueError("no quote has an implied volatility; check the inputs")
    vega = _vegas(quotes, iv_mkt)
    atm = float(np.nanmedian(iv_mkt))

    def resid(x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                p = model_prices(Model.from_free(structure, x), quotes, n_terms)
            except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
                return np.full(len(quotes), np.sqrt(_PENALTY / len(quotes)))
        r = (p - quotes.prices) / vega
        return np.where(np.isfinite(r), r, np.sqrt(_PENALTY / len(quotes)))

    x0 = Model(
        structure, initial if initial is not None else initial_params(structure, atm)
    ).to_free()
    starts = [x0] + [x0 + rng.normal(scale=0.7, size=x0.size) for _ in range(max(n_starts, 1) - 1)]
    best: Optional[optimize.OptimizeResult] = None
    for s in starts:
        if np.sum(resid(s) ** 2) >= _PENALTY:
            continue
        with warnings.catch_warnings():
            # finite differences straddling the penalty wall produce inf - inf
            warnings.simplefilter("ignore", RuntimeWarning)
            res = optimize.least_squares(resid, s, method="trf", max_nfev=maxiter, x_scale="jac")
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise RuntimeError(f"no admissible starting point for {structure.key()}")

    model = Model.from_free(structure, best.x)
    prices = model_prices(model, quotes, n_terms)
    iv_model = quotes.implied_vols(prices)
    err = iv_model - iv_mkt
    ok = np.isfinite(err)
    n = int(ok.sum())
    sse = float(np.sum(err[ok] ** 2))
    extras = {
        "iv_rmse": float(np.sqrt(sse / max(n, 1))),
        "iv_max_error": float(np.max(np.abs(err[ok]))) if n else float("nan"),
        "price_rel_rmse": float(np.sqrt(np.mean(((prices - quotes.prices) / quotes.prices) ** 2))),
    }
    for T, idx in quotes.maturity_groups():
        e = err[idx][np.isfinite(err[idx])]
        extras[f"iv_rmse@{T:g}"] = float(np.sqrt(np.mean(e**2))) if e.size else float("nan")
    loglik = -0.5 * n * np.log(max(sse, 1e-300) / max(n, 1))
    return FitResult(
        model=model,
        method="options",
        objective=float(2 * best.cost),
        n_obs=n,
        loglik=float(loglik),
        converged=bool(best.success),
        extras=extras,
    )


def discover_from_options(
    quotes: OptionQuotes,
    *,
    grammar: Optional[Grammar] = None,
    criterion: Literal["bic", "aic"] = "bic",
    search: Literal["beam", "exhaustive"] = "beam",
    beam_width: int = 3,
    max_terms: Optional[int] = None,
    n_starts: int = 2,
    seed: int = 0,
    n_terms: int = 1024,
    exclude_degenerate: bool = False,
) -> DiscoveryReport:
    """Search ``grammar`` for the risk-neutral structures that best price ``quotes``.

    ``exclude_degenerate`` works as in :func:`~spxa.discovery.search.discover`.
    The vanishing-summand check matters especially here: several components
    can fit a smile almost equally well.
    """
    from spxa.discovery.search import _warm_start

    grammar = grammar or Grammar()
    top = grammar.max_terms if max_terms is None else max_terms
    if criterion not in ("bic", "aic"):
        raise ValueError(f"criterion must be 'bic' or 'aic', got {criterion!r}")
    rng = np.random.default_rng(seed)
    fitted: dict[str, FitResult] = {}
    notes: list[str] = []
    atm = float(np.nanmedian(quotes.implied_vols()))

    def fit(s: Node, initial: Optional[list[tuple[float, ...]]] = None) -> Optional[FitResult]:
        key = s.key()
        if key in fitted:
            return fitted[key]
        try:
            r = fit_options(
                s,
                quotes,
                n_starts=n_starts if initial is None else 1,
                rng=np.random.default_rng(rng.integers(2**32)),
                initial=initial,
                n_terms=n_terms,
            )
        except (RuntimeError, ValueError) as exc:
            notes.append(f"{key}: fit failed ({exc})")
            return None
        fitted[key] = r
        return r

    def score(r: FitResult) -> float:
        v = r.bic if criterion == "bic" else r.aic
        return float(v) if v is not None else np.inf

    if search == "exhaustive":
        for s in grammar.structures(top):
            fit(s)
    elif search == "beam":
        level = [r for r in (fit(a) for a in grammar.atoms()) if r is not None]
        for _ in range(2, top + 1):
            parents = sorted(level, key=score)[:beam_width]
            level = []
            for parent in parents:
                for child in grammar.extensions(parent.model.structure):
                    c = canonical(child)
                    r = fit(c, _warm_start(parent, c, 0.5 * atm))
                    if r is not None:
                        level.append(r)
    else:
        raise ValueError(f"search must be 'beam' or 'exhaustive', got {search!r}")
    if not fitted:
        raise RuntimeError("no structure could be fitted to the quotes")
    ranked = sorted(fitted.values(), key=score)
    if len(quotes.maturity_groups()) > 1:
        b = ranked[0].extras
        per = {float(k[8:]): v for k, v in b.items() if k.startswith("iv_rmse@") and np.isfinite(v)}
        worst, best_t = max(per, key=per.__getitem__), min(per, key=per.__getitem__)
        if per[worst] > 0.005 and per[worst] > 2.5 * per[best_t]:
            notes.append(
                f"pricing errors depend on maturity (IV RMSE {100 * per[worst]:.2f} vol points "
                f"at T = {worst:g} vs {100 * per[best_t]:.2f} at T = {best_t:g}): a Lévy model's "
                "smile flattens like T^-1/2, which these quotes do not follow"
            )
    ranked, excluded, screen_notes = screen_degenerate(ranked, exclude_degenerate, score, criterion)
    return DiscoveryReport(
        candidates=ranked,
        summary=None,
        criterion=criterion,
        method="options",
        tolerance=0.0,
        warnings=notes + screen_notes,
        excluded=excluded,
    )
