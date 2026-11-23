r"""Mathematical Audit for Ornstein-Uhlenbeck and Vasicek diffusions."""

import numpy as np
from scipy import stats

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_characteristic_function,
    verify_moment,
)
from spxa.zoo.beyond.ornstein_uhlenbeck import OrnsteinUhlenbeck


def audit_ornstein_uhlenbeck() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="OrnsteinUhlenbeck")
    rng = np.random.default_rng(701)

    theta = 1.5  # Mean reversion speed kappa
    mu = 0.08  # Long run mean
    sigma = 0.25  # Volatility
    x0 = 0.04
    t = 1.5
    n_samples = 40_000

    ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, x0=x0)

    # 1. Exact conditional moments
    theo_mean = ou.conditional_mean(t=t, x0=x0)
    theo_var = ou.conditional_variance(t=t)

    # Simulation
    paths = ou.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]

    verify_moment(result, "Conditional_Mean_E[X_t|x0]", theoretical=theo_mean, samples=term)
    verify_moment(result, "Conditional_Var[X_t|x0]", theoretical=theo_var, samples=term)

    # 2. Stationary moments (at long horizon T = 15.0)
    t_long = 15.0
    paths_long = ou.simulate(n_steps=100, n_paths=n_samples, T=t_long, rng=rng)
    term_long = paths_long[:, -1]
    stat_mean = ou.stationary_mean
    stat_var = ou.stationary_variance

    verify_moment(result, "Stationary_Mean", theoretical=stat_mean, samples=term_long)
    verify_moment(result, "Stationary_Variance", theoretical=stat_var, samples=term_long)

    # 3. Kolmogorov-Smirnov test against stationary Gaussian distribution
    # Pass the frozen normal's CDF: SciPy 1.18 maps "norm" to special.ndtr,
    # which rejects loc/scale given through args=.
    ks_res = stats.kstest(term_long, stats.norm(stat_mean, np.sqrt(stat_var)).cdf)
    result.add_check(
        name="Stationary_KS_Uniformity",
        theoretical=0.0,
        empirical=float(ks_res.statistic),
        passed=bool(ks_res.pvalue >= 0.01),
        details=f"KS p-value={ks_res.pvalue:.4f}",
    )

    # 4. Characteristic function
    def cf(u_grid):
        return np.array([ou.char_func(u, t=t, x0=x0) for u in u_grid])

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    # 5. Vasicek zero coupon bond pricing Riccati check
    # P(t, T) = E^Q[exp(- int_t^T r_s ds)]
    # Bond price must be between 0 and 1 for positive rates
    tau = 2.0
    p_bond = float(ou.bond_price(tau=tau, r0=x0))
    result.add_check(
        name="BondPrice_Range",
        theoretical=p_bond,
        empirical=p_bond,
        passed=bool(0.0 < p_bond < 1.5),
        details=f"P(0, 2) = {p_bond:.6f}",
    )

    return result


if __name__ == "__main__":
    res = audit_ornstein_uhlenbeck()
    print(res.summary())
