"""Makes a two parameter DerivKit Fisher forecast for coffee cooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from getdist import plots as getdist_plots

from percolator_fish.data import make_coffee_data
from percolator_fish.fisher import diagonal_data_covariance, derivkit_fisher_forecast
from percolator_fish.model import coffee_temperature

DK_RED = "#f21901"
DK_BLUE = "#0173b2"

DERIVKIT_KWARGS: dict[str, Any] = {
    "method": "finite",
    "stepsize": 1e-2,
    "num_points": 5,
    "extrapolation": "ridders",
    "levels": 4,
}


@dataclass(frozen=True)
class CoffeeExperiment:
    """Defines one assumed coffee observing setup.

    Attributes:
        label: Name used in printed output and plot legends.
        time_min: Measurement times in minutes.
        sigma_temperature: Assumed independent temperature uncertainty in
            degrees Celsius for each measurement.
    """

    label: str
    time_min: np.ndarray
    sigma_temperature: float


def main() -> None:
    """Runs the two parameter coffee cooling Fisher forecast."""
    output_dir = Path("plots_output")
    output_dir.mkdir(exist_ok=True)

    data = make_coffee_data()
    theta0 = data["theta0"]

    experiments = [
        CoffeeExperiment(
            label="Short experiment",
            time_min=np.linspace(0.0, 40.0, 12),
            sigma_temperature=1.0,
        ),
        CoffeeExperiment(
            label="Long experiment",
            time_min=np.linspace(0.0, 1000.0, 30),
            sigma_temperature=1.0,
        ),
    ]

    forecasts = []

    for experiment in experiments:

        def model(
            theta: np.ndarray,
            time_min: np.ndarray = experiment.time_min,
        ) -> np.ndarray:
            """Returns the coffee temperature data vector."""
            return coffee_temperature(theta, time_min)

        cov = diagonal_data_covariance(
            experiment.time_min.size,
            experiment.sigma_temperature,
        )

        forecast = derivkit_fisher_forecast(
            model=model,
            theta0=theta0,
            cov=cov,
            names=["T0", "tau"],
            labels=[r"T_0", r"\tau"],
            **DERIVKIT_KWARGS,
        )

        forecasts.append(forecast)

    plotter = getdist_plots.get_subplot_plotter(width_inch=4.0)
    plotter.settings.linewidth_contour = 1.5
    plotter.settings.linewidth = 1.5

    plotter.triangle_plot(
        [forecast.gaussian for forecast in forecasts],
        params=forecasts[0].names,
        legend_labels=[experiment.label for experiment in experiments],
        filled=[False, False],
        contour_colors=[DK_RED, DK_BLUE],
        contour_lws=[1.5, 1.5],
        contour_ls=["-", "-"],
    )

    plotter.export(str(output_dir / "coffee_two_param_forecast.png"))

    for experiment, forecast in zip(experiments, forecasts, strict=True):
        print(f"\n{experiment.label}")
        print("Parameter names:", forecast.names)
        print("Fisher matrix:")
        print(forecast.fisher)

    print("\nSaved plots_output/coffee_two_param_forecast.png")


if __name__ == "__main__":
    main()
