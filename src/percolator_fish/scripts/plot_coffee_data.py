"""Makes synthetic coffee cooling data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from percolator_fish.data import make_coffee_data

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
        color=DK_RED,
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



def main() -> None:
    """Makes, saves, and plots synthetic coffee cooling data."""
    output_dir = Path("data_output")
    plot_dir = Path("plots_output")
    output_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    data = make_coffee_data()
    np.savez(output_dir / "coffee_data.npz", **data)

    plot_coffee_data(
        data["time_min"],
        data["observed_temperature"],
        data["true_temperature"],
        data["sigma_temperature"],
        plot_dir / "coffee_data.png",
    )

    print("Saved data_output/coffee_data.npz")
    print("Saved plots_output/coffee_data.png")


if __name__ == "__main__":
    main()
