"""Self-similarity flags only on strictly self-similar processes."""

from __future__ import annotations

import numpy as np
import pytest

from spxa.zoo.levy import AlphaStable, BrownianMotion, Drift


def _self_similar(X, H: float) -> bool:  # type: ignore[no-untyped-def]
    u = np.array([0.3, 1.7, -2.2])
    return all(
        np.allclose(X.char_func(u, t), X.char_func(t**H * u, 1.0), rtol=1e-12, atol=1e-14)
        for t in (0.5, 3.0)
    )


def test_brownian_motion_with_drift_is_not_self_similar() -> None:
    assert BrownianMotion(mu=0.0, sigma=0.7).properties.self_similarity_index == 0.5
    X = BrownianMotion(mu=0.3, sigma=0.7)
    assert X.properties.self_similarity_index is None
    assert not _self_similar(X, 0.5)


@pytest.mark.parametrize(
    "alpha, beta, mu, strictly",
    [
        (1.5, 0.3, 0.0, True),
        (1.5, 0.3, 0.2, False),
        (0.7, -1.0, 0.0, True),
        (1.0, 0.0, 0.4, True),
        (1.0, 0.5, 0.0, False),
        (2.0, 0.0, 0.0, True),
    ],
)
def test_alpha_stable(alpha: float, beta: float, mu: float, strictly: bool) -> None:
    X = AlphaStable(alpha=alpha, beta=beta, sigma=0.8, mu=mu)
    H = X.properties.self_similarity_index
    assert (H is not None) == strictly
    assert _self_similar(X, 1.0 / alpha) == strictly
    if strictly:
        assert H == pytest.approx(1.0 / alpha)


def test_drift_is_self_similar_with_index_one() -> None:
    assert _self_similar(Drift(0.4), 1.0)
