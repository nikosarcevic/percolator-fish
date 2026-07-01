"""Makes synthetic coffee cooling data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from percolator_fish.data import make_coffee_data
from percolator_fish.plots import plot_coffee_data


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