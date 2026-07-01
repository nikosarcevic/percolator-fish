"""Makes a two parameter DerivKit Fisher forecast for coffee cooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from getdist import plots as getdist_plots

from percolator_fish.data import make_coffee_data
from percolator_fish.fisher import diagonal_data_covariance, derivkit_fisher_forecast
from percolator_fish.model import coffee_temperature

DK_RED = "#f21901"



def main() -> None:
    output_dir = Path("plots_output")
    output_dir.mkdir(exist_ok=True)

    data = make_coffee_data()
    time_min = data["time_min"]
    theta0 = data["theta0"]
    sigma_temperature = float(data["sigma_temperature"])
    cov = diagonal_data_covariance(time_min.size, sigma_temperature)


    def model(theta: np.ndarray) -> np.ndarray:
        """Returns the coffee temperature data vector."""
        return coffee_temperature(theta, time_min)


    forecast = derivkit_fisher_forecast(
        model=model,
        theta0=theta0,
        cov=cov,
        names=["T0", "tau"],
        labels=[r"T_0", r"\tau"],
        stepsize=1e-2,
    )

    plotter = getdist_plots.get_subplot_plotter(width_inch=4.0)
    plotter.settings.linewidth_contour = 1.5
    plotter.settings.linewidth = 1.5
    plotter.triangle_plot(
        [forecast.gaussian],
        params=forecast.names,
        filled=[False],
        contour_colors=[DK_RED],
        contour_lws=[1.5],
        contour_ls=["-"],
    )
    plotter.export(str(output_dir / "coffee_two_param_forecast.png"))

    print("Parameter names:", forecast.names)
    print("Fisher matrix:")
    print(forecast.fisher)
    print("Saved plots_output/coffee_two_param_forecast.png")


if __name__ == "__main__":
    main()
