"""Simulation backends for spxa."""

from spxa.sim.euler import (
    euler_maruyama,
    euler_maruyama_autonomous,
    geometric_levy,
    convergence_order_estimate,
)
from spxa.sim.exact import (
    sample_brownian,
    sample_gamma,
    sample_inverse_gaussian,
    sample_vg,
    sample_nig,
    sample_alpha_stable,
    sample_compound_poisson,
    sample_tempered_stable,
    increments_to_paths,
)
from spxa.sim.bridge import (
    brownian_bridge,
    gamma_bridge,
    brownian_bridge_interpolate,
    first_passage_time_bm,
    first_passage_time_exact_bm,
)

__all__ = [
    "euler_maruyama",
    "euler_maruyama_autonomous",
    "geometric_levy",
    "convergence_order_estimate",
    "sample_brownian",
    "sample_gamma",
    "sample_inverse_gaussian",
    "sample_vg",
    "sample_nig",
    "sample_alpha_stable",
    "sample_compound_poisson",
    "sample_tempered_stable",
    "increments_to_paths",
    "brownian_bridge",
    "gamma_bridge",
    "brownian_bridge_interpolate",
    "first_passage_time_bm",
    "first_passage_time_exact_bm",
]
