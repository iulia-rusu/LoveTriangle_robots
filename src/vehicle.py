from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np

from .geometry import wrap_angle


@dataclass
class VehicleState:
    x: float
    y: float
    heading: float
    left_motor: float = 0.0
    right_motor: float = 0.0
    forward_speed: float = 0.0
    angular_velocity: float = 0.0
    prev_forward_speed: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


class DifferentialDriveVehicle:
    """Simple differential-drive vehicle model."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = self._random_state()

    def reset(self) -> VehicleState:
        self.state = self._random_state()
        return self.state

    def _random_state(self) -> VehicleState:
        vcfg = self.cfg["vehicle"]
        arena = self.cfg["arena"]
        radius = float(vcfg["radius"])
        half_w = float(arena["width"]) / 2.0
        half_h = float(arena["height"]) / 2.0
        return VehicleState(
            x=random.uniform(-half_w + radius, half_w - radius),
            y=random.uniform(-half_h + radius, half_h - radius),
            heading=random.uniform(-math.pi, math.pi),
        )

    def update(self, left_motor: float, right_motor: float, dt: float) -> VehicleState:
        vcfg = self.cfg["vehicle"]
        clip = float(vcfg["motor_clip"])
        max_speed = float(vcfg["max_linear_speed"])
        wheel_base = float(vcfg["wheel_base"])

       

        st = self.state
        st.prev_forward_speed = st.forward_speed
        st.left_motor = left_motor
        st.right_motor = right_motor

        v_left = left_motor 
        v_right = right_motor 

        st.forward_speed = 0.5 * (v_left + v_right)
        st.angular_velocity = (v_right - v_left) / max(wheel_base, 1e-6)

        st.heading = wrap_angle(st.heading + st.angular_velocity * dt)
        st.x += st.forward_speed * math.cos(st.heading) * dt
        st.y += st.forward_speed * math.sin(st.heading) * dt

        return st
