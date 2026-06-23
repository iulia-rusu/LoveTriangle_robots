from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def stimulus_positions(condition: str, t: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return red and green stimulus positions for a named condition."""
    if condition == "red_only":
        return np.array([56.0, 0.0], dtype=np.float32), None

    if condition == "green_only":
        return None, np.array([12.0, 0.0], dtype=np.float32)

    if condition == "separated":
        red = np.array([50, 20], dtype=np.float32)
        green = np.array([-80, -40], dtype=np.float32)
        return red, green

    if condition == "blocking":
        green = np.array([56.0, 0.0], dtype=np.float32)
        
        red = np.array([14.0, 24.0 * math.sin(0.7 * t)], dtype=np.float32)
        return red, green

    if condition == "crossing":
        red = np.array([56.0 * math.cos(0.25 * t), 36.0 * math.sin(0.25 * t)], dtype=np.float32)
        green = np.array([48.0 * math.cos(0.25 * t + math.pi), 28.0 * math.sin(0.25 * t + math.pi)], dtype=np.float32)
        return red, green

    if condition == "orbit":
        red = np.array([60.0 * math.cos(0.35 * t), 70.0 * math.sin(0.35 * t)], dtype=np.float32)
        green = np.array([35.0 * math.cos(0.65 * t + 1.5), 35.0 * math.sin(0.65 * t + 1.5)], dtype=np.float32)
        return red, green
    
    raise ValueError(f"Unknown condition: {condition}")
