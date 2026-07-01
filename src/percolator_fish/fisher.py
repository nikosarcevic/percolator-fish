"""DerivKit Fisher forecast helpers for the coffee cooling example."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from derivkit import ForecastKit
from getdist.gaussian_mixtures import GaussianND


FORECASTKIT_RESERVED_KWARGS = {
    "order",
}


@dataclass(frozen=True)
class FisherForecast:
    """Stores a Fisher forecast and the metadata needed for plotting.

    Attributes:
        fisher: Fisher information matrix for the forecast parameters.
        names: Internal parameter names passed to GetDist.
        labels: Plot labels for the forecast parameters.
        gaussian: GetDist Gaussian approximation built from the Fisher matrix.
    """

    fisher: np.ndarray
    names: list[str]
    labels: list[str]
    gaussian: GaussianND


def diagonal_data_covariance(
    n_data: int,
    sigma: float,
) -> np.ndarray:
    """Builds a diagonal data covariance matrix with constant variance.

    Args:
        n_data: Number of data points in the data vector.
        sigma: Standard deviation of each independent data point.

    Returns:
        Diagonal covariance matrix with shape ``(n_data, n_data)``.
    """
    return np.eye(n_data) * sigma**2


def add_gaussian_priors(
    fisher: np.ndarray,
    prior_sigma: np.ndarray,
) -> np.ndarray:
    """Adds independent Gaussian parameter priors to a Fisher matrix.

    Each independent Gaussian prior adds ``1 / sigma**2`` to the diagonal
    Fisher entry for the corresponding parameter.

    Args:
        fisher: Fisher matrix from the data likelihood.
        prior_sigma: Prior standard deviation for each parameter.

    Returns:
        Fisher matrix including the independent Gaussian prior information.
    """
    return fisher + np.diag(1.0 / prior_sigma**2)


def derivkit_fisher_forecast(
    *,
    model: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    cov: np.ndarray,
    names: Sequence[str],
    labels: Sequence[str],
    label: str = "DerivKit Fisher",
    prior_sigma: np.ndarray | None = None,
    **derivkit_kwargs: Any,
) -> FisherForecast:
    """Computes a DerivKit Fisher forecast for a model data vector.

    The forecast evaluates derivatives of ``model`` around the fiducial
    parameter vector ``theta0``, combines those derivatives with the data
    covariance matrix, optionally adds independent Gaussian priors, and converts
    the result into a GetDist Gaussian for plotting.

    DerivKit keyword arguments are passed directly to ``ForecastKit.fisher``.
    This wrapper deliberately does not add default derivative options because
    different DerivKit derivative methods accept different keyword arguments.
    For example, finite differences can use ``stepsize``, while adaptive or
    local polynomial methods may not.

    Args:
        model: Function that maps a parameter vector to a model data vector.
        theta0: Fiducial parameter vector where the forecast is evaluated.
        cov: Data covariance matrix for the model data vector.
        names: Internal parameter names used by GetDist.
        labels: Plot labels used by GetDist.
        label: Legend label for the GetDist Gaussian.
        prior_sigma: Optional independent Gaussian prior widths.
        **derivkit_kwargs: Keyword arguments forwarded to
            ``ForecastKit.fisher``. Do not pass ``order`` because
            ``ForecastKit`` sets the derivative order internally.

    Returns:
        Fisher forecast container with the Fisher matrix, parameter metadata,
        and GetDist Gaussian approximation.

    Raises:
        ValueError: If a keyword reserved by ``ForecastKit`` is passed through
            ``derivkit_kwargs``.
    """
    invalid_kwargs = FORECASTKIT_RESERVED_KWARGS.intersection(derivkit_kwargs)

    if invalid_kwargs:
        invalid = ", ".join(sorted(invalid_kwargs))
        msg = (
            "Do not pass these keyword arguments through "
            f"derivkit_fisher_forecast: {invalid}. ForecastKit sets them "
            "internally."
        )
        raise ValueError(msg)

    fk = ForecastKit(function=model, theta0=theta0, cov=cov)
    fisher = fk.fisher(**derivkit_kwargs)

    if prior_sigma is not None:
        fisher = add_gaussian_priors(fisher, prior_sigma)

    parameter_names = list(names)
    parameter_labels = list(labels)

    gaussian = fk.getdist_fisher_gaussian(
        fisher=fisher,
        names=parameter_names,
        labels=parameter_labels,
        label=label,
    )

    return FisherForecast(
        fisher=fisher,
        names=parameter_names,
        labels=parameter_labels,
        gaussian=gaussian,
    )
