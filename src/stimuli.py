from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def stimulus_positions(condition: str, t: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return red and green stimulus positions for a named condition."""
    if condition == "red_only":
        return np.array([2.8, 0.0], dtype=np.float32), None

    if condition == "green_only":
        return None, np.array([0.6, 0.0], dtype=np.float32)

    if condition == "separated":
        red = np.array([2.8, 1.2], dtype=np.float32)
        green = np.array([0.5, -1.4], dtype=np.float32)
        return red, green

    if condition == "blocking":
        red = np.array([2.8, 0.0], dtype=np.float32)
        # Green oscillates across the direct path to red.
        green = np.array([0.7, 1.2 * math.sin(0.7 * t)], dtype=np.float32)
        return red, green

    if condition == "crossing":
        red = np.array([2.8 * math.cos(0.25 * t), 1.8 * math.sin(0.25 * t)], dtype=np.float32)
        green = np.array([2.4 * math.cos(0.25 * t + math.pi), 1.4 * math.sin(0.25 * t + math.pi)], dtype=np.float32)
        return red, green

    if condition == "orbit":
        red = np.array([2.5 * math.cos(0.35 * t), 2.0 * math.sin(0.35 * t)], dtype=np.float32)
        green = np.array([1.2 * math.cos(0.65 * t + 1.5), 1.2 * math.sin(0.65 * t + 1.5)], dtype=np.float32)
        return red, green
    
    raise ValueError(f"Unknown condition: {condition}")
