"""Plotting helpers for the coffee Fisher demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DK_BLUE = "#3b9ab2"
DK_RED = "#f21901"


def plot_coffee_data(
    time_min: np.ndarray,
    observed_temperature: np.ndarray,
    true_temperature: np.ndarray,
    sigma_temperature: float,
    output_path: str | Path,
) -> None:
    """Plots synthetic coffee cooling data."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        time_min,
        observed_temperature,
        yerr=sigma_temperature,
        fmt="o",
        color=DK_BLUE,
        label="Noisy measurements",
    )
    ax.plot(
        time_min,
        true_temperature,
        color=DK_RED,
        linewidth=2.0,
        label="True cooling model",
    )

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Temperature [°C]")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
