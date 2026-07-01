"""Synthetic coffee cooling data."""

from __future__ import annotations

import numpy as np

from percolator_fish.model import coffee_temperature


def make_coffee_data(
    *,
    seed: int = 42,
    sigma_temperature: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Makes noisy synthetic coffee cooling data."""
    rng = np.random.default_rng(seed)

    time_min = np.arange(0.0, 61.0, 5.0)
    theta0 = np.array([90.0, 25.0])
    true_temperature = coffee_temperature(theta0, time_min)
    observed_temperature = true_temperature + rng.normal(
        0.0,
        sigma_temperature,
        size=time_min.size,
    )

    return {
        "time_min": time_min,
        "theta0": theta0,
        "true_temperature": true_temperature,
        "observed_temperature": observed_temperature,
        "sigma_temperature": sigma_temperature,
    }
