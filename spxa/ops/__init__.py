"""Operations on stochastic processes."""

from spxa.ops.addition import add, exactness_for_addition, sum_processes, triplet_add
from spxa.ops.integral import (
    StochasticIntegral,
    ito_isometry_variance,
    stochastic_convolution_char_func,
)
from spxa.ops.product import (
    ProductProcess,
    cumulants_to_raw_moments,
    multiply,
    raw_moments_to_cumulants,
)
from spxa.ops.scaling import negate, scale, subtract, triplet_scale
from spxa.ops.subordination import (
    bernstein_function,
    char_exp_subordinated,
    is_valid_subordination,
    subordinate,
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
