"""Operations on stochastic processes."""

from spxa.ops.addition import add, sum_processes, triplet_add, exactness_for_addition
from spxa.ops.scaling import scale, negate, subtract, triplet_scale
from spxa.ops.subordination import (
    subordinate,
    char_exp_subordinated,
    bernstein_function,
    is_valid_subordination,
)
from spxa.ops.product import multiply, ProductProcess, cumulants_to_raw_moments, raw_moments_to_cumulants
from spxa.ops.integral import (
    ito_isometry_variance,
    stochastic_convolution_char_func,
    StochasticIntegral,
)

__all__ = [
    "add",
    "sum_processes",
    "triplet_add",
    "exactness_for_addition",
    "scale",
    "negate",
    "subtract",
    "triplet_scale",
    "subordinate",
    "char_exp_subordinated",
    "bernstein_function",
    "is_valid_subordination",
    "multiply",
    "ProductProcess",
    "cumulants_to_raw_moments",
    "raw_moments_to_cumulants",
    "ito_isometry_variance",
    "stochastic_convolution_char_func",
    "StochasticIntegral",
]
