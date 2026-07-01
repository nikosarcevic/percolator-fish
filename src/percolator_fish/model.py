"""Coffee cooling models for DerivKit Fisher demos."""

from __future__ import annotations

import numpy as np


def coffee_temperature(
    theta: np.ndarray,
    time_min: np.ndarray,
    *,
    room_temperature: float = 22.0,
) -> np.ndarray:
    """Predicts coffee temperature using Newton cooling.

    Args:
        theta: Model parameters ``[initial_temperature, cooling_time]``.
        time_min: Measurement times in minutes.
        room_temperature: Fixed ambient room temperature in degrees Celsius.

    Returns:
        Coffee temperature in degrees Celsius.
    """
    initial_temperature, cooling_time = theta

    return room_temperature + (
        initial_temperature - room_temperature
    ) * np.exp(-time_min / cooling_time)


def advanced_coffee_temperature(
    theta: np.ndarray,
    time_min: np.ndarray,
) -> np.ndarray:
    """Predicts coffee temperature with intentionally coupled parameters."""
    initial_temperature, room_temperature, cooling_time, cup_factor = theta

    amplitude = initial_temperature - room_temperature
    effective_cooling_time = cooling_time * cup_factor

    return (
        room_temperature
        + amplitude * np.exp(-time_min / effective_cooling_time)
        + 0.08 * (initial_temperature - 90.0) * time_min / 60.0
        - 0.15 * (room_temperature - 22.0) * time_min / 60.0
        - 4.0 * (cup_factor - 1.0) * time_min / 60.0
    )