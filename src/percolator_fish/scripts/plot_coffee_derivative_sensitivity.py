"""Plots a DerivKit five-point derivative of the coffee-cooling model.

This figure illustrates the model derivative entering a Fisher forecast. For
one model parameter, the derivative is not a single number. It is a vector with
one component for every observation time, describing how the entire predicted
data vector responds when that parameter changes.

For a parameter ``theta_i``, DerivKit evaluates the model using a centered
five-point finite-difference stencil,

    dT(t) / dtheta_i
        ≈ [
              T(t; theta_i - 2h)
            - 8 T(t; theta_i - h)
            + 8 T(t; theta_i + h)
            - T(t; theta_i + 2h)
          ] / (12h).

The upper panel shows the fiducial model and the four neighboring model
evaluations used by the five-point stencil:

    theta_i - 2h,
    theta_i - h,
    theta_i,
    theta_i + h,
    theta_i + 2h.

The lower panel shows the derivative vector returned by
``DerivativeKit.differentiate`` using ``method="finite"`` and
``num_points=5``.

The step used here is intentionally large enough for the neighboring curves to
be distinguishable in a presentation. It is a visualization choice, not a
recommended production step size. Reliable Fisher forecasts require derivative
stability to be checked across reasonable numerical prescriptions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from derivkit import DerivativeKit

from percolator_fish.data import make_coffee_data
from percolator_fish.model import coffee_temperature


REPO_ROOT = Path(__file__).resolve().parents[3]

DK_RED = "#f21901"
DARK_GRAY = "#555555"
LIGHT_GRAY = "#999999"

# Choose which parameter derivative to illustrate.
#
# "T0"  shows dT(t) / dT0
# "tau" shows dT(t) / dtau
PLOT_PARAMETER = "T0"

PARAMETER_INFO: dict[str, dict[str, Any]] = {
    "T0": {
        "index": 0,
        "symbol": r"T_0",
        "derivative_label": r"$\partial T(t) / \partial T_0$",
        "derivative_units": r"$^\circ\mathrm{C}/^\circ\mathrm{C}$",
    },
    "tau": {
        "index": 1,
        "symbol": r"\tau",
        "derivative_label": r"$\partial T(t) / \partial \tau$",
        "derivative_units": r"$^\circ\mathrm{C}/\mathrm{min}$",
    },
}

# This is deliberately large enough for the neighboring curves to be visible
# in a presentation. It is not a recommended production derivative step.
PERTURBATION_FRACTION = 0.05


def make_one_parameter_model(
    *,
    theta0: np.ndarray,
    parameter_index: int,
    time_min: np.ndarray,
) -> Callable[[float], np.ndarray]:
    """Builds a scalar-parameter model for DerivativeKit.

    All model parameters except the selected parameter are held fixed at their
    fiducial values.

    Args:
        theta0: Full fiducial parameter vector.
        parameter_index: Index of the parameter to vary.
        time_min: Times at which to evaluate the coffee model.

    Returns:
        Function mapping the selected scalar parameter value to the complete
        coffee-temperature data vector.
    """

    def model(parameter_value: float) -> np.ndarray:
        """Returns the coffee data vector for one parameter value."""
        theta = theta0.copy()
        theta[parameter_index] = parameter_value
        return coffee_temperature(theta, time_min)

    return model


def main() -> None:
    """Creates the five-point derivative figure for the presentation."""
    output_dir = REPO_ROOT / "plots_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = make_coffee_data()
    theta0 = np.asarray(data["theta0"], dtype=float)

    parameter = PARAMETER_INFO[PLOT_PARAMETER]
    parameter_index = int(parameter["index"])
    parameter_symbol = str(parameter["symbol"])

    time_min = np.linspace(0.0, 120.0, 300)

    fiducial_value = float(theta0[parameter_index])
    step = PERTURBATION_FRACTION * abs(fiducial_value)

    if step == 0.0:
        raise ValueError(
            f"Cannot define a fractional perturbation for {PLOT_PARAMETER} "
            "because its fiducial value is zero."
        )

    if PLOT_PARAMETER == "tau" and fiducial_value - 2.0 * step <= 0.0:
        raise ValueError(
            "The lower five-point stencil evaluation gives a non-positive tau."
        )

    model = make_one_parameter_model(
        theta0=theta0,
        parameter_index=parameter_index,
        time_min=time_min,
    )

    stencil_values = {
        "minus_2h": fiducial_value - 2.0 * step,
        "minus_h": fiducial_value - step,
        "fiducial": fiducial_value,
        "plus_h": fiducial_value + step,
        "plus_2h": fiducial_value + 2.0 * step,
    }

    model_minus_2h = model(stencil_values["minus_2h"])
    model_minus_h = model(stencil_values["minus_h"])
    model_fiducial = model(stencil_values["fiducial"])
    model_plus_h = model(stencil_values["plus_h"])
    model_plus_2h = model(stencil_values["plus_2h"])

    derivative_kit = DerivativeKit(
        function=model,
        x0=fiducial_value,
    )

    derivative = np.asarray(
        derivative_kit.differentiate(
            method="finite",
            order=1,
            stepsize=step,
            num_points=5,
        ),
        dtype=float,
    )

    fig = plt.figure(figsize=(9.0, 7.0))

    grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[2.0, 1.0],
        hspace=0.12,
    )

    model_ax = fig.add_subplot(grid[0])
    derivative_ax = fig.add_subplot(grid[1], sharex=model_ax)

    model_ax.plot(
        time_min,
        model_minus_2h,
        color=LIGHT_GRAY,
        linestyle=":",
        linewidth=2.0,
        label=rf"${parameter_symbol}-2h$",
    )

    model_ax.plot(
        time_min,
        model_minus_h,
        color=LIGHT_GRAY,
        linestyle="--",
        linewidth=2.5,
        label=rf"${parameter_symbol}-h$",
    )

    model_ax.plot(
        time_min,
        model_fiducial,
        color="black",
        linestyle="-",
        linewidth=3.0,
        label="Fiducial model",
    )

    model_ax.plot(
        time_min,
        model_plus_h,
        color=DK_RED,
        linestyle="--",
        linewidth=2.5,
        label=rf"${parameter_symbol}+h$",
    )

    model_ax.plot(
        time_min,
        model_plus_2h,
        color=DK_RED,
        linestyle=":",
        linewidth=2.0,
        label=rf"${parameter_symbol}+2h$",
    )

    derivative_ax.plot(
        time_min,
        derivative,
        color=DK_RED,
        linewidth=3.0,
    )

    derivative_ax.axhline(
        0.0,
        color=DARK_GRAY,
        linewidth=1.0,
        alpha=0.5,
    )

    model_ax.set_ylabel(
        r"Temperature [$^\circ$C]",
        fontsize=18,
    )

    derivative_ax.set_xlabel(
        "Elapsed time [min]",
        fontsize=18,
    )

    derivative_ax.set_ylabel(
        f"{parameter['derivative_label']}\n"
        f"[{parameter['derivative_units']}]",
        fontsize=16,
    )

    model_ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=14,
        ncol=2,
    )

    model_ax.tick_params(
        axis="both",
        labelsize=15,
    )

    derivative_ax.tick_params(
        axis="both",
        labelsize=15,
    )

    model_ax.tick_params(
        axis="x",
        labelbottom=False,
    )

    for axis in (model_ax, derivative_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0.0)

    output_path = output_dir / f"coffee_derivative_sensitivity_{PLOT_PARAMETER}.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Parameter: {PLOT_PARAMETER}")
    print(f"Fiducial value: {fiducial_value:.6g}")
    print(f"Finite-difference step h: {step:.6g}")
    print("Derivative method: finite")
    print("Number of stencil points: 5")
    print(f"Derivative shape: {derivative.shape}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
