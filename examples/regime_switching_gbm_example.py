#!/usr/bin/env python3
"""
Example: Regime-Switching Geometric Brownian Motion (RS-GBM).

Demonstrates:
  1. Initialization of a two-state Markov-switching economy (Bull vs Bear / Low-Vol vs High-Vol).
  2. Stationary distribution and continuous-time Markov transition probabilities.
  3. Closed-form matrix exponential expectations, second moments, and conditional volatility.
  4. Exact continuous characteristic function evaluation.
  5. Gil-Pelaez European call and put option pricing with put-call parity verification.
  6. Coupled Markov-jump and asset price path simulation.
"""

import numpy as np

from spxa import RegimeSwitchingGBM


def main() -> None:
    print("=" * 72)
    print("Regime-Switching Geometric Brownian Motion (RS-GBM) Demonstration")
    print("=" * 72)

    # 1. Initialize 2-state economy:
    # State 0: Low volatility Bull market:  μ = +10%, σ = 15%
    # State 1: High volatility Bear market: μ = -6%,  σ = 35%
    # Transition rates:
    # λ_{01} = 0.40 (annual rate of entering recession)
    # λ_{10} = 1.20 (annual rate of exiting recession, mean duration 10 months)
    q_rates = np.array(
        [
            [-0.40, 0.40],
            [1.20, -1.20],
        ]
    )

    rs_gbm = RegimeSwitchingGBM(
        mu=[0.10, -0.06],
        sigma=[0.15, 0.35],
        transition_rates=q_rates,
        s0=100.0,
        initial_regime=0,
    )

    print(f"\nModel Configuration: {rs_gbm}")
    print(f"  Regime 0 (Bull/Low-Vol):  μ = {rs_gbm.mu[0]:+.2%}, σ = {rs_gbm.sigma[0]:.2%}")
    print(f"  Regime 1 (Bear/High-Vol): μ = {rs_gbm.mu[1]:+.2%}, σ = {rs_gbm.sigma[1]:.2%}")
    stat_pi = rs_gbm.stationary_distribution
    print(f"  Stationary Probabilities: π_0 = {stat_pi[0]:.2%}, π_1 = {stat_pi[1]:.2%}")

    # 2. Transition matrix at horizon T = 1.0
    tau = 1.0
    p_tau = rs_gbm.transition_probabilities(t=tau)
    print(f"\nTransition Probabilities P(T = {tau:.1f} yr):")
    print(f"  P(stay in Bull | start Bull): {p_tau[0, 0]:.4f}")
    print(f"  P(switch to Bear | start Bull): {p_tau[0, 1]:.4f}")
    print(f"  P(stay in Bear | start Bear): {p_tau[1, 1]:.4f}")
    print(f"  P(switch to Bull | start Bear): {p_tau[1, 0]:.4f}")

    # 3. Exact Matrix Exponential Moments
    mean_1 = rs_gbm.conditional_mean(t=tau)
    sec_mom = rs_gbm.conditional_second_moment(t=tau)
    var_1 = rs_gbm.conditional_variance(t=tau)
    std_1 = rs_gbm.conditional_std(t=tau)

    print(f"\nConditional Moments at horizon T = {tau:.1f} yr (Initial State: Bull):")
    print(f"  Expected Price E[S_T]:        ${mean_1:.4f}")
    print(f"  Second Moment E[S_T²]:        ${sec_mom:.2f}")
    print(f"  Conditional Variance:          {var_1:.4f} (Std Dev: ${std_1:.4f})")

    # 4. European Options via Gil-Pelaez Fourier Inversion
    strike = 105.0
    r_discount = 0.05
    call = rs_gbm.call_price(strike=strike, time_to_maturity=tau, discount_rate=r_discount)
    put = rs_gbm.put_price(strike=strike, time_to_maturity=tau, discount_rate=r_discount)
    parity_disc_forward = np.exp(-r_discount * tau) * (mean_1 - strike)

    print(
        f"\nEuropean Options (Strike = ${strike:.2f}, Horizon = {tau:.1f} yr, r = {r_discount:.2%}):"
    )
    print(f"  Call Price:                   ${call:.4f}")
    print(f"  Put Price:                    ${put:.4f}")
    print(f"  Call - Put:                   ${call - put:.4f}")
    print(f"  Parity RHS e^(-rT)(E[S_T]-K): ${parity_disc_forward:.4f}")
    print(f"  Parity Absolute Error:         {abs(call - put - parity_disc_forward):.2e}")

    # 5. Simulation
    n_paths = 50000
    n_steps = 100
    print(f"\nSimulating {n_paths:,} joint paths ({n_steps} steps over T = {tau:.1f} yr)...")
    s_paths, reg_paths = rs_gbm.simulate_joint(
        n_steps=n_steps, time_horizon=tau, n_paths=n_paths, rng=np.random.default_rng(42)
    )

    s_term = s_paths[:, -1]
    emp_mean = np.mean(s_term)
    emp_var = np.var(s_term)
    terminal_bear_freq = np.mean(reg_paths[:, -1] == 1)

    print(f"  Empirical Mean:               ${emp_mean:.4f} (Theoretical: ${mean_1:.4f})")
    print(f"  Empirical Variance:            {emp_var:.4f} (Theoretical: {var_1:.4f})")
    print(
        f"  Terminal Bear Fraction:        {terminal_bear_freq:.2%} (Theoretical: {p_tau[0, 1]:.2%})"
    )

    print("\nDemonstration completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
