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
        self.rng = np.random.default_rng()
        self.step_idx = 0
        self.time = 0.0
        self.red_pos: np.ndarray | None = None
        self.green_pos: np.ndarray | None = None
        self.last_info: StepInfo | None = None
        self.action_space = None  
        self.state = None  # Initialize the state attribute
        self.reward = 0.0  # Initialize the reward attribute

    @property
    
    
    
    def state_dim(self) -> int:
        return 9


    def reset(
    self,
    random_start: bool = False,
    rng: np.random.Generator | None = None,
    start_margin: float | None = None):
        self.action_space = self.make_action()
        self.step_idx = 0
        self.time = 0.0
        self.vehicle.reset()
        if random_start:
            rng = rng or np.random.default_rng()
            arena_w = float(self.cfg["arena"]["width"])
            arena_h = float(self.cfg["arena"]["height"])
            radius = float(self.cfg["vehicle"]["radius"])
            margin = radius if start_margin is None else float(start_margin)

            self.vehicle.state.x = float(rng.uniform(-arena_w / 2 + margin, arena_w / 2 - margin))
            self.vehicle.state.y = float(rng.uniform(-arena_h / 2 + margin, arena_h / 2 - margin))
            self.vehicle.state.heading = float(rng.uniform(-np.pi, np.pi))
        self.red_pos, self.green_pos = stimulus_positions(self.condition, self.time)
        
        reason = "reset"
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
        self.state = self.build_state()
        return self.state,self.last_info

    
    def make_action(self, duration = 0.5):
        actions = list()
        for i in range(-int(self.cfg["vehicle"]["max_linear_speed"]),int(self.cfg["vehicle"]["max_linear_speed"])+1,20):
            for j in range(-int(self.cfg["vehicle"]["max_linear_speed"]),int(self.cfg["vehicle"]["max_linear_speed"])+1,20):
                action = [i, j]
                actions.append(action)
        return actions
        



    def step(self, action: np.ndarray | None = None, apply_action_noise: bool = False, action_noise_std: float = 0.1):
        """Advance the simulation by one step using the given action.

        apply_action_noise: if True, perturbs the resolved left/right motor
        speeds with multiplicative Gaussian noise (std = action_noise_std * |speed|)
        before applying them, then clips back to the valid speed range. Models
        real-world motor calibration error for sim-to-real robustness training;
        off by default so eval/rollout calls are unaffected.
        """
        self.step_idx += 1

        dt = float(self.cfg["simulation"]["dt"])
        self.time += dt

        if action is None:
            left_motor, right_motor = 0.0, 0.0
        else:
            # Convert PyTorch tensor to NumPy
            if hasattr(action, "detach"):
                action = action.detach().cpu().numpy()

            action = np.asarray(action).squeeze()

            # Case 1: DQN action index, e.g. tensor([[110]])
            if action.ndim == 0:
                action_idx = int(action.item())
                left_motor, right_motor = self.action_space[action_idx]

            # Case 2: real motor command, e.g. [-30, 10]
            elif action.size == 2:
                left_motor, right_motor = action.astype(float)

            else:
                raise ValueError(f"Invalid action shape/value: {action}")

            left_motor = float(left_motor)
            right_motor = float(right_motor)

        if apply_action_noise:
            max_speed = float(self.cfg["vehicle"]["max_linear_speed"])
            left_motor += self.rng.normal(0.0, action_noise_std * abs(left_motor))
            right_motor += self.rng.normal(0.0, action_noise_std * abs(right_motor))
            left_motor = float(np.clip(left_motor, -max_speed, max_speed))
            right_motor = float(np.clip(right_motor, -max_speed, max_speed))

        prev_x = self.last_info.vehicle_x if self.last_info is not None else self.vehicle.state.x
        prev_y = self.last_info.vehicle_y if self.last_info is not None else self.vehicle.state.y

        # update vehicle here
        self.vehicle.update(left_motor=left_motor, right_motor=right_motor, dt=dt)

        # then build state using previous position
        self.state = self.build_state(prev_x=prev_x, prev_y=prev_y)


        self.red_pos, self.green_pos = stimulus_positions(
            self.condition,
            self.time
        )

        done, reason = self._termination_reason()
        self.reward = self.reward_function()

        self.last_info = StepInfo(
            step=self.step_idx,
            time=self.time,
            red_pos=self.red_pos,
            green_pos=self.green_pos,
            left_motor=self.vehicle.state.left_motor,
            right_motor=self.vehicle.state.right_motor,
            done_reason=reason,
            vehicle_x=self.vehicle.state.x,
            vehicle_y=self.vehicle.state.y,
        )

        return self.build_state(), self.reward, done
    
    def build_state(self, prev_x: float | None = None, prev_y: float | None = None) -> np.ndarray:
        
        x = self.vehicle.state.x
        y = self.vehicle.state.y

        arena_w = float(self.cfg["arena"]["width"])
        arena_h = float(self.cfg["arena"]["height"])
        half_w = arena_w / 2.0
        half_h = arena_h / 2.0

        # centred coordinates: x is in roughly [-half_w, half_w]
        x_norm = x / half_w
        y_norm = y / half_h

        dt = float(self.cfg["simulation"]["dt"])

        if prev_x is None or prev_y is None:
            agent_vel = 0.0
        else:
            agent_vel = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2) / dt

        norm_vel = agent_vel / float(self.cfg["vehicle"]["max_linear_speed"])

        heading_theta = self.vehicle.state.heading
        sin_theta = np.sin(heading_theta)
        cos_theta = np.cos(heading_theta)

        red_x, red_y = self.red_pos
        green_x, green_y = self.green_pos

        norm_red_x = red_x / half_w
        norm_red_y = red_y / half_h
        norm_green_x = green_x / half_w
        norm_green_y = green_y / half_h

        return np.array(
            [
                x_norm,
                y_norm,
                sin_theta,
                cos_theta,
                norm_vel,
                norm_green_x,
                norm_green_y,
                norm_red_x,
                norm_red_y,
            ],
            dtype=np.float32,
        )
    
    def reward_function(self):
        """"Decay reward function considers the distance between the agent and the green robot, as well as the distance to the arena boundaries. The reward is higher when the agent is closer to the green robot and further from the boundaries."""
        reward = 0.0
        # Adjust this factor to control the decay rate
        red_pos, green_pos = self.red_pos, self.green_pos
        vehicle_pos = np.array([self.vehicle.state.x, self.vehicle.state.y])
        distance_to_green = np.linalg.norm(vehicle_pos - green_pos)
        arena_width = float(self.cfg["arena"]["width"])
        arena_height = float(self.cfg["arena"]["height"])
        half_w = arena_width / 2.0
        half_h = arena_height / 2.0
        distance_to_boundaries = min(
            vehicle_pos[0] + half_w,   # distance from left wall
            half_w - vehicle_pos[0],   # distance from right wall
            vehicle_pos[1] + half_h,   # distance from bottom wall
            half_h - vehicle_pos[1],   # distance from top wall
        )
        distance_to_red = np.linalg.norm(vehicle_pos - red_pos)
        if distance_to_green < 10:  # Threshold for being very close to the green robot
            reward += 100.0  # Reward for being very close to the green robot
        
        elif distance_to_boundaries < 5:  # Threshold for being very close to the boundaries
            reward -= 5.0 * 1/(distance_to_boundaries)  # Penalty grows the closer/further past the wall the agent gets
        
        if distance_to_red < 10:  # Threshold for being very close to the red robot
            reward -= 100.0  # Penalty for being very close to the red robot
        reward-= 0.1 * distance_to_green  # Encourage getting closer to the green robot
        # reward -= 0.05 * distance_to_boundaries  # Encourage staying away from the
        reward += 0.05 * distance_to_red  # Encourage staying away from the red robot
        # reward -= 0.01 * self.time  # Small penalty for time to encourage faster completion
        
        return reward
    
    
        
        
    def _termination_reason(self) -> tuple[bool, str | None]:
        st = self.vehicle.state
        arena = self.cfg["arena"]
        radius = float(self.cfg["vehicle"]["radius"])
        max_steps = int(self.cfg["simulation"]["max_steps"])

        if not inside_box(st.position, float(arena["width"]), float(arena["height"]), margin=radius):
            return True, "wall_collision"

    
        if self.step_idx >= max_steps:
            return True, "timeout"
        
        if self.red_pos is not None and np.linalg.norm(st.position - self.red_pos) < radius:
            return True, "red_collision"
        
        if self.green_pos is not None and np.linalg.norm(st.position - self.green_pos) < radius:
            return True, "green_collision"

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
