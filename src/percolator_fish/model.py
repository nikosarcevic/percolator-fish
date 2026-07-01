"""Coffee cooling models used in the DerivKit Fisher forecast demos."""

from __future__ import annotations

import numpy as np


def coffee_temperature(
    theta: np.ndarray,
    time_min: np.ndarray,
    *,
    room_temperature: float = 22.0,
) -> np.ndarray:
    """Evaluates a Newton cooling model for a cup of coffee.

    The model assumes that the coffee temperature relaxes exponentially toward
    a fixed room temperature,

        T(t) = T_room + (T_0 - T_room) exp(-t / tau),

    where ``T_0`` is the initial coffee temperature and ``tau`` is the cooling
    time scale.

    Args:
        theta: Parameter vector ``[initial_temperature, cooling_time]``.
            ``initial_temperature`` is measured in degrees Celsius and
            ``cooling_time`` is measured in minutes.
        time_min: Measurement times in minutes.
        room_temperature: Fixed ambient room temperature in degrees Celsius.

    Returns:
        Model coffee temperatures in degrees Celsius evaluated at ``time_min``.
    """
    initial_temperature, cooling_time = theta

    return room_temperature + (
        initial_temperature - room_temperature
    ) * np.exp(-time_min / cooling_time)


def advanced_coffee_temperature(
    theta: np.ndarray,
    time_min: np.ndarray,
) -> np.ndarray:
    """Evaluates a four parameter coffee cooling model with degeneracies.

    This model extends pure Newton cooling by allowing the ambient room
    temperature and the cup properties to vary. It is designed as a forecasting
    demo: the parameters affect the temperature curve in overlapping ways, so
    the Fisher matrix produces visible parameter degeneracies.

    The parameter vector is

        ``[initial_temperature, room_temperature, cooling_time, cup_factor]``.

    Args:
        theta: Parameter vector for the advanced cooling model.

            ``initial_temperature``:
                Initial coffee temperature in degrees Celsius. This controls
                the starting temperature and the overall cooling amplitude.

            ``room_temperature``:
                Ambient room temperature in degrees Celsius. This controls the
                long time temperature that the coffee relaxes toward.

            ``cooling_time``:
                Exponential cooling time scale in minutes. Larger values make
                the coffee cool more slowly.

            ``cup_factor``:
                Dimensionless factor that rescales the effective cooling time.
                Values larger than one mimic a more insulating cup, while values
                smaller than one mimic a cup that lets the coffee cool faster.

        time_min: Measurement times in minutes.

    Returns:
        Model coffee temperatures in degrees Celsius evaluated at ``time_min``.
    """
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
