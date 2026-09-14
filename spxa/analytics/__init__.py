"""Analytics module — divergences, Fourier pricing, cumulant tools."""

from spxa.analytics.cumulants import cumulant_table, excess_kurtosis, skewness
from spxa.analytics.divergence import (
    hellinger_distance,
    kl_divergence_mc,
    l2_char_func_distance,
    wasserstein2_distance,
)
from spxa.analytics.fourier import (
    european_call_price,
    european_put_price,
    implied_volatility,
)

__all__ = [
    "cumulant_table",
    "excess_kurtosis",
    "skewness",
    "hellinger_distance",
    "kl_divergence_mc",
    "l2_char_func_distance",
    "wasserstein2_distance",
    "european_call_price",
    "european_put_price",
    "implied_volatility",
]
