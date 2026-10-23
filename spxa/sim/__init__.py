"""Simulation backends for spxa."""

from spxa.sim.bridge import (
    brownian_bridge,
    brownian_bridge_interpolate,
    first_passage_time_bm,
    first_passage_time_exact_bm,
    gamma_bridge,
)
from spxa.sim.euler import (
    convergence_order_estimate,
    euler_maruyama,
    euler_maruyama_autonomous,
    geometric_levy,
)
from spxa.sim.exact import (
    increments_to_paths,
    sample_alpha_stable,
    sample_brownian,
    sample_cev,
    sample_cir,
    sample_compound_poisson,
    sample_gamma,
    sample_heston,
    sample_inverse_gaussian,
    sample_merton,
    sample_nig,
    sample_ou,
    sample_tempered_stable,
    sample_vg,
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
    "sample_cir",
    "sample_ou",
    "sample_heston",
    "sample_merton",
    "sample_cev",
    "increments_to_paths",
    "brownian_bridge",
    "gamma_bridge",
    "brownian_bridge_interpolate",
    "first_passage_time_bm",
    "first_passage_time_exact_bm",
]
