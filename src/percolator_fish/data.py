"""Synthetic coffee cooling data."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from percolator_fish.model import coffee_temperature


class CoffeeData(TypedDict):
    """Synthetic coffee cooling data products."""

    time_min: np.ndarray
    theta0: np.ndarray
    true_temperature: np.ndarray
    observed_temperature: np.ndarray
    sigma_temperature: np.ndarray
    noise_fraction: float


def make_coffee_data(
    *,
    seed: int = 42,
    noise_fraction: float = 0.05,
) -> CoffeeData:
    """Makes noisy synthetic coffee cooling data.

    The synthetic data are generated from the fiducial coffee cooling model.
    Independent Gaussian noise is added to each temperature sample, with a
    standard deviation equal to a fixed fraction of the true temperature.

    For example, ``noise_fraction=0.05`` means that each data point has
    5 percent temperature noise.

    Args:
        seed: Random seed used to generate the noise realization.
        noise_fraction: Fractional Gaussian temperature noise. For example,
            ``0.05`` corresponds to 5 percent noise.

    Returns:
        Dictionary containing the time samples, fiducial parameters, noiseless
        temperatures, noisy observed temperatures, and per point temperature
        uncertainties.
    """
    rng = np.random.default_rng(seed)

    time_min = np.arange(0.0, 61.0, 5.0)
    theta0 = np.array([90.0, 25.0])

    true_temperature = coffee_temperature(theta0, time_min)
    sigma_temperature = noise_fraction * true_temperature

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
        "noise_fraction": noise_fraction,
    }
