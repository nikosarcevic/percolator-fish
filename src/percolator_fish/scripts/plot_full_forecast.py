"""Makes a multi parameter DerivKit Fisher triangle plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from getdist import plots as getdist_plots

from percolator_fish.fisher import diagonal_data_covariance, derivkit_fisher_forecast
from percolator_fish.model import advanced_coffee_temperature

DK_RED = "#f21901"

DERIVKIT_KWARGS: dict[str, Any] = {
    "method": "finite",
    "stepsize": 1e-3,
    "num_points": 5,
    "extrapolation": "ridders",
    "levels": 4,
}


def main() -> None:
    """Runs the multi parameter coffee cooling Fisher forecast."""
    output_dir = Path("plots_output")
    output_dir.mkdir(exist_ok=True)

    time_min = np.arange(10.0, 61.0, 10.0)
    sigma_temperature = 2.0
    cov = diagonal_data_covariance(time_min.size, sigma_temperature)

    theta0 = np.array([90.0, 22.0, 25.0, 1.0])

    def model(theta: np.ndarray) -> np.ndarray:
        """Returns the advanced coffee temperature data vector."""
        return advanced_coffee_temperature(theta, time_min)

    forecast = derivkit_fisher_forecast(
        model=model,
        theta0=theta0,
        cov=cov,
        names=["T0", "Troom", "tau", "cup"],
        labels=[r"T_0", r"T_{\rm room}", r"\tau", r"c_{\rm cup}"],
        prior_sigma=np.array([80.0, 80.0, 80.0, 4.0]),
        **DERIVKIT_KWARGS,
    )

    lw = 2
    fs = 20

    plotter = getdist_plots.get_subplot_plotter(width_inch=8.0)
    plotter.settings.linewidth_contour = lw
    plotter.settings.linewidth = lw
    plotter.settings.axes_labelsize = fs
    plotter.settings.axes_fontsize = fs

    plotter.triangle_plot(
        [forecast.gaussian],
        params=forecast.names,
        filled=[False],
        contour_colors=[DK_RED],
        contour_lws=[lw],
        contour_ls=["-"],
    )

    figname = "coffee_full_forecast.png"
    plotter.export(str(output_dir / figname))

    print("Parameter names:", forecast.names)
    print("Fisher matrix:")
    print(forecast.fisher)
    print(f"Saved plots_output/{figname}")


if __name__ == "__main__":
    main()
