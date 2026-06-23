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
        self.action_space = self.make_action()

    @property
    
    
    
    def state_dim(self) -> int:
        return 18


    def reset(self):
        self.vehicle.reset()
        self.red_pos, self.green_pos = stimulus_positions(self.condition, self.time)
        self.step_idx = 0
        self.time = 0.0
        return self.last_info
    
    def make_action(duration = 0.5):
        actions = []
        for i in range(-50,51):
            for j in range(-50,51):
                action = [i, j, duration]
                actions.append(action)
        return actions
        



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
    
    def build_state(self) -> np.ndarray:
        x = self.vehicle.x # need to normalize this
        y = self.vehicle.y
        x_norm = x/self.cfg["arena"]["width"]
        y_norm = y/self.cfg["arena"]["height"]
        dt = self.cfg["simulation"]["dt"]
        last_x= self.last_info.vehicle_x
        last_y = self.last_info.vehicle_y
        agent_vel = np.sqrt((x-last_x)**2 + (y-last_y)**2)/dt
        heading_theta = self.vehicle.heading
        sin_theta = np.sin(heading_theta)
        cos_theta = np.cos(heading_theta)
        agent_vel = self.last_info.
        red_x, red_y = self.red_pos
        green_x, green_y = self.green_pos


        return np.array([x_norm, y_norm, sin_theta, cos_theta, agent_vel, green_x, green_y,  red_x, red_y])
        
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
