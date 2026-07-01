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
            time_min=np.linspace(0.0, 40.0, 20),
            sigma_temperature=1.0,
        ),
        CoffeeExperiment(
            label="Long experiment",
            time_min=np.linspace(0.0, 1000.0, 100),
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

    lw = 2
    fs = 20

    plotter = getdist_plots.get_subplot_plotter(width_inch=8.0)
    plotter.settings.linewidth_contour = lw
    plotter.settings.linewidth = lw
    plotter.settings.axes_labelsize = fs
    plotter.settings.axes_fontsize = fs
    plotter.settings.legend_fontsize = fs
    plotter.settings.figure_legend_frame = False

    plotter.triangle_plot(
        [forecast.gaussian for forecast in forecasts],
        params=forecasts[0].names,
        legend_labels=[experiment.label for experiment in experiments],
        filled=[False, False],
        contour_colors=[DK_RED, "k"],
        contour_lws=[lw, lw],
        contour_ls=["-", "-"],
    )

    figname = "coffee_simple_forecast.png"
    plotter.export(str(output_dir / figname))

    for experiment, forecast in zip(experiments, forecasts, strict=True):
        print(f"\n{experiment.label}")
        print("Parameter names:", forecast.names)
        print("Fisher matrix:")
        print(forecast.fisher)

    print(f"\nSaved plots_output/{figname}")


if __name__ == "__main__":
    main()
