"""DerivKit Fisher forecast helpers for coffee cooling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from derivkit import ForecastKit
from getdist.gaussian_mixtures import GaussianND


@dataclass(frozen=True)
class FisherForecast:
    """Container for a Fisher forecast and plotting metadata."""

    fisher: np.ndarray
    names: list[str]
    labels: list[str]
    gaussian: GaussianND


def diagonal_data_covariance(
    n_data: int,
    sigma: float,
) -> np.ndarray:
    """Builds a diagonal Gaussian data covariance matrix."""
    return np.eye(n_data) * sigma**2


def add_gaussian_priors(
    fisher: np.ndarray,
    prior_sigma: np.ndarray,
) -> np.ndarray:
    """Adds independent Gaussian priors to a Fisher matrix."""
    return fisher + np.diag(1.0 / prior_sigma**2)


def derivkit_fisher_forecast(
    *,
    model: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    cov: np.ndarray,
    names: Sequence[str],
    labels: Sequence[str],
    label: str = "DerivKit Fisher",
    stepsize: float = 1e-2,
    prior_sigma: np.ndarray | None = None,
) -> FisherForecast:
    """Computes a DerivKit Fisher forecast and GetDist Gaussian."""
    fk = ForecastKit(function=model, theta0=theta0, cov=cov)

    fisher = fk.fisher(
        method="finite",
        stepsize=stepsize,
        num_points=5,
        extrapolation="ridders",
        levels=4,
    )

    if prior_sigma is not None:
        fisher = add_gaussian_priors(fisher, prior_sigma)

    gaussian = fk.getdist_fisher_gaussian(
        fisher=fisher,
        names=list(names),
        labels=list(labels),
        label=label,
    )

    return FisherForecast(
        fisher=fisher,
        names=list(names),
        labels=list(labels),
        gaussian=gaussian,
    )
