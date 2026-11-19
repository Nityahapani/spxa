"""
Example 07: Copula-Coupled Stochastic Processes & Multi-Asset Derivatives.

Demonstrates:
1. Coupling univariate stochastic processes (GBM, Merton Jump-Diffusion) via copulas.
2. Simulating coupled multi-asset sample paths under asymmetric Clayton dependence.
3. Pricing multi-asset exotic derivatives: Basket Calls, Worst-Of Calls, Best-Of Calls, Spread Options.
4. Parameter calibration via Canonical Maximum Likelihood (CML) and Kendall's tau inversion.
5. Copula model selection using AIC/BIC information criteria across competing families.
6. Generating 3D copula probability surfaces and tail dependence diagnostic curves.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure spxa package is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spxa import (
    BrownianMotion,
    ClaytonCopula,
    CoupledProcess,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    MertonJumpDiffusion,
    StudentTCopula,
    plot_copula_cdf,
    plot_copula_pdf,
    plot_copula_samples,
    plot_tail_dependence_curves,
    select_best_copula,
)


def main() -> None:
    output_dir = Path("examples_output/copula")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("SPXA Copula Module: Multi-Asset Dependence & Process Coupling")
    print("=" * 70)

    # 1. Instantiate Copulas & Verify Analytical Properties
    print("\n1. Analytical Copula Properties:")
    clayton = ClaytonCopula(theta=2.5)
    gumbel = GumbelCopula(theta=2.5)
    student_t = StudentTCopula(df=4.0, corr=0.6)
    gaussian = GaussianCopula(corr=0.6)

    print(
        f"  Clayton (θ=2.5)   : Kendall's τ = {clayton.kendall_tau():.4f}, Tail Dep (λ_L, λ_U) = {clayton.tail_dependence()}"
    )
    print(
        f"  Gumbel (θ=2.5)    : Kendall's τ = {gumbel.kendall_tau():.4f}, Tail Dep (λ_L, λ_U) = {gumbel.tail_dependence()}"
    )
    print(
        f"  Student-t (ν=4,ρ=0.6): Kendall's τ = {student_t.kendall_tau():.4f}, Tail Dep (λ_L, λ_U) = {student_t.tail_dependence()}"
    )
    print(
        f"  Gaussian (ρ=0.6)  : Kendall's τ = {gaussian.kendall_tau():.4f}, Tail Dep (λ_L, λ_U) = {gaussian.tail_dependence()}"
    )

    # 2. Coupled Process Simulation
    print("\n2. Coupling Stochastic Processes via Sklar's Theorem:")
    asset1 = MertonJumpDiffusion(rate=1.2, jump_mean=-0.08, jump_std=0.12)
    asset2 = BrownianMotion(mu=0.06, sigma=0.22)

    coupled = CoupledProcess(processes=[asset1, asset2], copula=clayton)
    print(f"  Coupled System: {coupled}")

    n_paths = 20000
    T = 1.0
    r = 0.05
    strike = 1.0

    print(f"  Simulating {n_paths} joint paths over T={T}...")
    paths = coupled.simulate(n_steps=50, n_paths=n_paths, T=T, rng=np.random.default_rng(42))
    s1_terminal = paths[:, -1, 0]
    s2_terminal = paths[:, -1, 1]

    log_ret1 = np.log(s1_terminal / paths[:, 0, 0])
    log_ret2 = np.log(s2_terminal / paths[:, 0, 1])
    emp_corr = np.corrcoef(log_ret1, log_ret2)[0, 1]
    print(f"  Empirical Terminal Return Correlation: {emp_corr:.4f}")

    # 3. Multi-Asset Exotic Derivative Pricing
    print("\n3. Multi-Asset Exotic Derivative Valuation (Monte Carlo):")
    basket_price = coupled.basket_call_price(
        weights=[0.5, 0.5], strike=strike, T=T, r=r, n_paths=n_paths
    )
    worst_of_price = coupled.worst_of_call_price(strike=strike, T=T, r=r, n_paths=n_paths)
    best_of_price = coupled.best_of_call_price(strike=strike, T=T, r=r, n_paths=n_paths)
    spread_price = coupled.spread_option_price(strike=0.0, T=T, r=r, n_paths=n_paths)

    print(f"  Worst-Of Call (min(S1, S2) - K)+ : {worst_of_price:.5f}")
    print(f"  Basket Call   (0.5 S1 + 0.5 S2 - K)+: {basket_price:.5f}")
    print(f"  Best-Of Call  (max(S1, S2) - K)+ : {best_of_price:.5f}")
    print(f"  Spread Option (S1 - S2)+         : {spread_price:.5f}")
    print(f"  Arbitrage Ordering Holds: {worst_of_price <= basket_price <= best_of_price}")

    # 4. Calibration & Model Selection
    print("\n4. Model Selection & AIC/BIC Evaluation on Simulated Market Data:")
    # Extract pseudo-observations from market returns
    market_data = np.column_stack((log_ret1, log_ret2))
    best_cop, report = select_best_copula(
        market_data[:2000],
        candidate_families=[
            GaussianCopula,
            StudentTCopula,
            ClaytonCopula,
            GumbelCopula,
            FrankCopula,
        ],
    )
    print(f"  Best Fitting Copula Family: {report['best_model']} (AIC = {report['best_aic']:.2f})")
    print("  Model Comparison Table:")
    for model_name, info in report["models"].items():
        print(
            f"    - {model_name:18s}: Log-Lik = {info['log_likelihood']:8.2f}, AIC = {info['aic']:8.2f}, Params = {info['parameters']}"
        )

    # 5. Visualizations
    print("\n5. Generating Visualizations...")
    pdf_fig, _ = plot_copula_pdf(clayton, save_path=str(output_dir / "clayton_pdf_surface.png"))
    cdf_fig, _ = plot_copula_cdf(clayton, save_path=str(output_dir / "clayton_cdf_surface.png"))
    scat_fig, _ = plot_copula_samples(
        clayton, n_samples=1500, save_path=str(output_dir / "clayton_scatter_spaces.png")
    )

    tail_fig, _ = plot_tail_dependence_curves(
        {
            "Clayton (θ=2.5)": clayton,
            "Gumbel (θ=2.5)": gumbel,
            "Student-t (ν=4, ρ=0.6)": student_t,
            "Gaussian (ρ=0.6)": gaussian,
        },
        save_path=str(output_dir / "tail_dependence_comparison.png"),
    )

    print(f"  Saved all visual plots to {output_dir.resolve()}/")
    print("\n✓ Copula module demonstration completed successfully!")


if __name__ == "__main__":
    main()
