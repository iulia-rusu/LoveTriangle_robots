from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def unit_from_angle(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


def left_axis(theta: float) -> np.ndarray:
    return np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)


def right_axis(theta: float) -> np.ndarray:
    return -left_axis(theta)


def ray_distance_to_box(
    origin: np.ndarray,
    direction: np.ndarray,
    width: float,
    height: float,
) -> float:
    """Distance from a point to the boundary of a centred rectangular arena.

    The arena spans [-width/2, width/2] x [-height/2, height/2].
    """
    x_min, x_max = -width / 2.0, width / 2.0
    y_min, y_max = -height / 2.0, height / 2.0
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])

    candidates: list[float] = []
    eps = 1e-9

    if abs(dx) > eps:
        for x_wall in (x_min, x_max):
            t = (x_wall - ox) / dx
            y = oy + t * dy
            if t >= 0 and y_min - 1e-6 <= y <= y_max + 1e-6:
                candidates.append(t)

    if abs(dy) > eps:
        for y_wall in (y_min, y_max):
            t = (y_wall - oy) / dy
            x = ox + t * dx
            if t >= 0 and x_min - 1e-6 <= x <= x_max + 1e-6:
                candidates.append(t)

    if not candidates:
        return float("inf")
    return float(min(candidates))


def inside_box(pos: np.ndarray, width: float, height: float, margin: float = 0.0) -> bool:
    return (
        -width / 2.0 + margin <= pos[0] <= width / 2.0 - margin
        and -height / 2.0 + margin <= pos[1] <= height / 2.0 - margin
    )
