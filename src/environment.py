from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .vehicle import DifferentialDriveVehicle
from .geometry import inside_box
from .stimuli import stimulus_positions




@dataclass
class StepInfo:
    step: int
    time: float
    vehicle_x: float
    vehicle_y: float
    red_pos: np.ndarray | None
    green_pos: np.ndarray | None
    left_motor: float
    right_motor: float
    done_reason: str | None


class BraitenbergEnv:
    """Simulation environment"""

    def __init__(self, cfg: dict, condition: str | None = None):
        self.cfg = cfg
        self.condition = condition or cfg["simulation"]["condition"]
        self.vehicle = DifferentialDriveVehicle(cfg)
        self.step_idx = 0
        self.time = 0.0
        self.red_pos: np.ndarray | None = None
        self.green_pos: np.ndarray | None = None
        self.last_info: StepInfo | None = None

    @property
    def state_dim(self) -> int:
        return 18


    def reset(self):
        self.vehicle.reset()
        self.red_pos, self.green_pos = stimulus_positions(self.condition, self.time)
        self.step_idx = 0
        self.time = 0.0
        return 0


    def step(self, action: np.ndarray | None = None):
        """Advance the simulation by one step using the given action."""
        self.step_idx += 1
        dt = float(self.cfg["simulation"]["dt"])
        self.time += dt
        self.vehicle.update(left_motor=0.0, right_motor=0.0, dt=dt)  # No action for now
        self.red_pos, self.green_pos = stimulus_positions(self.condition, self.time)
        done, reason = self._termination_reason()
        self.last_info = StepInfo(
            step=self.step_idx,
            time=self.time,
            red_pos=self.red_pos,
            green_pos=self.green_pos,
            left_motor=self.vehicle.state.left_motor,
            right_motor=self.vehicle.state.right_motor,
            done_reason=reason,
            vehicle_x=self.vehicle.state.x,
            vehicle_y=self.vehicle.state.y

        )
        
        return None, 0.0, done, self.last_info
        
    def _termination_reason(self) -> tuple[bool, str | None]:
        st = self.vehicle.state
        arena = self.cfg["arena"]
        radius = float(self.cfg["vehicle"]["radius"])
        max_steps = int(self.cfg["simulation"]["max_steps"])

        if not inside_box(st.position, float(arena["width"]), float(arena["height"]), margin=radius):
            return True, "wall_collision"

    
        if self.step_idx >= max_steps:
            return True, "timeout"

        return False, None

    def current_log_row(self, reward: float | None = None) -> dict:
        st = self.vehicle.state
        info = self.last_info
        row = {
            "step": self.step_idx,
            "time": self.time,
            "vehicle_x": st.x,
            "vehicle_y": st.y,
            "heading": st.heading,

            "green_x": np.nan if self.green_pos is None else float(self.green_pos[0]),
            "green_y": np.nan if self.green_pos is None else float(self.green_pos[1]),
            "reward": np.nan if reward is None else reward,
        }
        if info is not None:
            row.update({
                "left_motor": info.left_motor,
                "right_motor": info.right_motor,
            })
        return row
