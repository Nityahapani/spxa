"""
Markov-modulated Poisson process demo.

Run with: python examples/mmpp_demo.py
"""

import numpy as np

from spxa.zoo.beyond.mmpp import MarkovModulatedPoissonProcess

# Two background regimes: quiet (rate 4) and busy (rate 10).
# The chain leaves "quiet" at rate 1 and leaves "busy" at rate 2.
Q = [[-1.0, 1.0], [2.0, -2.0]]
rates = [4.0, 10.0]

pi = MarkovModulatedPoissonProcess(Q, rates).stationary_distribution()
p = MarkovModulatedPoissonProcess(Q, rates, initial_probabilities=pi)

print(f"process            : {p}")
print(f"exactness          : {p.exactness.name}")
print(f"stationary law     : {np.round(pi, 4)}")
print(f"mean intensity     : {p.mean_intensity:.4f}")

t = 2.0
mean, var = p.count_moments(t)
print(f"E[N_{t:g}], Var[N_{t:g}]: {mean:.4f}, {var:.4f}  (Fano factor {var / mean:.3f} > 1)")
print("cumulants of N_1   :", {k: round(v, 4) for k, v in p.cumulants(order=4).items()})

paths = p.simulate(n_steps=50, n_paths=5000, T=t, rng=np.random.default_rng(0))
print(f"simulated mean/var : {paths[:, -1].mean():.4f}, {paths[:, -1].var():.4f}")

events = p.simulate_events(T=t, rng=np.random.default_rng(1))
print(
    f"one sample path    : {events.arrival_count} arrivals, "
    f"{events.transition_times.size - 1} regime switches"
)
