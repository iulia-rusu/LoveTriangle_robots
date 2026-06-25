from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import deque, namedtuple
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=int(capacity))

    def push(self, *args: Any) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, int(batch_size))

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(int(n_observations), int(hidden_dim))
        self.layer2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.layer3 = nn.Linear(int(hidden_dim), int(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        memory_size: int = 10_000,
        eps_start: float = 0.9,
        eps_end: float = 0.01,
        eps_decay: int = 50_000,
        hidden_dim: int = 128,
    ):
        self.n_observations = int(n_observations)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = int(eps_decay)
        self.steps_done = 0

        self.policy_net = DQN(self.n_observations, self.n_actions, self.hidden_dim).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions, self.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=float(lr), amsgrad=True)
        self.memory = ReplayMemory(memory_size)

    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -float(self.steps_done) / float(self.eps_decay)
        )

    def select_action(self, state: torch.Tensor, explore: bool = True) -> torch.Tensor:
        eps_threshold = self.epsilon() if explore else 0.0
        if explore:
            self.steps_done += 1

        if (not explore) or random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

        action_idx = random.randrange(self.n_actions)
        return torch.tensor([[action_idx]], device=self.device, dtype=torch.long)

    def optimise_model(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=self.device,
            dtype=torch.bool,
        )

        non_final_next_states = None
        if non_final_mask.any():
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_next_states is not None:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = reward_batch + self.gamma * next_state_values
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.soft_update_target()
        return float(loss.item())

    def soft_update_target(self) -> None:
        target_state = self.target_net.state_dict()
        policy_state = self.policy_net.state_dict()
        for key in policy_state:
            target_state[key] = policy_state[key] * self.tau + target_state[key] * (1.0 - self.tau)
        self.target_net.load_state_dict(target_state)


@dataclass
class EpisodeStart:
    x: float | None = None
    y: float | None = None
    heading: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {"start_x": self.x, "start_y": self.y, "start_heading": self.heading}


def get_device(prefer_cpu: bool = False) -> torch.device:
    if prefer_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_state_tensor(state: np.ndarray | list[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    try:
        if re.search(r"[.eE]", text):
            return float(text)
        return int(text)
    except ValueError:
        return text


def set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    out = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, value = item.split("=", 1)
        set_nested(out, key.strip(), parse_scalar(value))
    return out


def resolve_condition(cfg: dict[str, Any], condition: str | None) -> str | None:
    resolved = condition
    if resolved is None:
        resolved = cfg.get("condition")
    if resolved is None:
        resolved = cfg.get("simulation", {}).get("condition")
    if resolved is not None:
        cfg.setdefault("simulation", {})["condition"] = resolved
    return resolved


def get_action_space(env: Any) -> list[Any]:
    if hasattr(env, "action_space"):
        return list(env.action_space)
    if hasattr(env, "make_action"):
        return list(env.make_action())
    raise AttributeError("Environment needs either env.action_space or env.make_action().")


def decode_action(action_idx: torch.Tensor, action_space: list[Any], mode: str) -> Any:
    """Return the object passed to env.step.

    mode='index': pass the DQN action index tensor, e.g. tensor([[3]]).
    mode='motor': pass the corresponding motor pair from action_space, e.g. np.array([1, -1]).
    """
    if mode not in {"index", "motor"}:
        raise ValueError("action mode must be 'index' or 'motor'")
    if mode == "index":
        return action_idx
    idx = int(action_idx.item())
    return np.asarray(action_space[idx], dtype=np.float32)


def normalise_step_result(result: Any) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    if not isinstance(result, tuple):
        raise TypeError("env.step(...) must return a tuple.")
    if len(result) == 3:
        observation, reward, terminated = result
        return observation, float(reward), bool(terminated), {}
    if len(result) == 4:
        observation, reward, terminated, info = result
        return observation, float(reward), bool(terminated), info or {}
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated or truncated), info or {}
    raise ValueError(f"Unsupported env.step return length: {len(result)}")


def step_env(env: Any, action_idx: torch.Tensor, action_space: list[Any], action_mode: str) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    env_action = decode_action(action_idx, action_space, action_mode)
    return normalise_step_result(env.step(env_action,apply_action_noise=True,
            action_noise_std=0.1))


def sample_episode_start(
    cfg: dict[str, Any],
    rng: np.random.Generator,
    margin: float = 0.0,
    random_heading: bool = True,
) -> EpisodeStart:
    half_w = float(cfg["arena"]["width"]) / 2.0
    half_h = float(cfg["arena"]["height"]) / 2.0
    margin = float(margin)
    margin_x = min(margin, max(0.0, half_w - 1e-6))
    margin_y = min(margin, max(0.0, half_h - 1e-6))

    x = float(rng.uniform(-half_w + margin_x, half_w - margin_x))
    y = float(rng.uniform(-half_h + margin_y, half_h - margin_y))
    if random_heading:
        heading = float(rng.uniform(-math.pi, math.pi))
    else:
        heading = float(cfg.get("vehicle", {}).get("start_heading", 0.0))
    return EpisodeStart(x=x, y=y, heading=heading)


def write_start_to_cfg(cfg: dict[str, Any], start: EpisodeStart) -> None:
    cfg.setdefault("vehicle", {})["start_x"] = float(start.x)
    cfg.setdefault("vehicle", {})["start_y"] = float(start.y)
    cfg.setdefault("vehicle", {})["start_heading"] = float(start.heading)


def force_vehicle_pose(env: Any, start: EpisodeStart) -> None:
    """Best-effort update for envs whose reset() caches the initial pose.

    Most versions of the project read cfg['vehicle']['start_*'] inside reset().
    This function also tries to update the vehicle state directly, so random
    starts continue to work if the vehicle object has already been created.
    """
    if start.x is None or start.y is None or start.heading is None:
        return

    vehicle = getattr(env, "vehicle", None)
    if vehicle is None:
        return

    candidates = [vehicle]
    if hasattr(vehicle, "state"):
        candidates.append(vehicle.state)

    for obj in candidates:
        for attr, value in (("x", start.x), ("y", start.y), ("heading", start.heading)):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, float(value))
                except Exception:
                    pass
        if hasattr(obj, "position"):
            try:
                obj.position = np.array([float(start.x), float(start.y)], dtype=float)
            except Exception:
                try:
                    obj.position[:] = [float(start.x), float(start.y)]
                except Exception:
                    pass


def reset_env(
    env: Any,
    cfg: dict[str, Any],
    rng: np.random.Generator | None = None,
    random_start: bool = False,
    start_margin: float = 0.0,
    random_heading: bool = True,
) -> tuple[np.ndarray, dict[str, Any], EpisodeStart]:
    start = EpisodeStart(
        x=float(cfg.get("vehicle", {}).get("start_x", 0.0)),
        y=float(cfg.get("vehicle", {}).get("start_y", 0.0)),
        heading=float(cfg.get("vehicle", {}).get("start_heading", 0.0)),
    )

    if random_start:
        if rng is None:
            rng = np.random.default_rng()
        start = sample_episode_start(cfg, rng, start_margin, random_heading)
        write_start_to_cfg(cfg, start)
        if hasattr(env, "cfg"):
            write_start_to_cfg(env.cfg, start)

    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        state, info = reset_result
    else:
        state, info = reset_result, {}

    if random_start:
        force_vehicle_pose(env, start)

    return state, info or {}, start


def get_vehicle_xy(env: Any) -> tuple[float, float]:
    vehicle = getattr(env, "vehicle", None)
    if vehicle is None:
        return float("nan"), float("nan")
    state = getattr(vehicle, "state", vehicle)
    if hasattr(state, "x") and hasattr(state, "y"):
        return float(state.x), float(state.y)
    if hasattr(state, "position"):
        pos = np.asarray(state.position)
        return float(pos[0]), float(pos[1])
    return float("nan"), float("nan")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    agent: DQNAgent,
    episode: int,
    episode_rewards: list[float],
    episode_durations: list[int],
    cfg: dict[str, Any],
    action_space: list[Any],
    best_reward: float,
    action_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": int(episode),
            "policy_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "steps_done": int(agent.steps_done),
            "n_observations": int(agent.n_observations),
            "n_actions": int(agent.n_actions),
            "hidden_dim": int(agent.hidden_dim),
            "episode_rewards": episode_rewards,
            "episode_durations": episode_durations,
            "best_reward": float(best_reward),
            "config": cfg,
            "action_space": action_space,
            "action_mode": action_mode,
        },
        path,
    )


def save_policy(path: Path, agent: DQNAgent, cfg: dict[str, Any], action_space: list[Any], action_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": agent.policy_net.state_dict(),
            "n_observations": int(agent.n_observations),
            "n_actions": int(agent.n_actions),
            "hidden_dim": int(agent.hidden_dim),
            "config": cfg,
            "action_space": action_space,
            "action_mode": action_mode,
        },
        path,
    )


def load_policy(path: Path, device: torch.device) -> tuple[DQN, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    hidden_dim = int(payload.get("hidden_dim", 128))
    net = DQN(payload["n_observations"], payload["n_actions"], hidden_dim=hidden_dim).to(device)
    net.load_state_dict(payload["policy_state_dict"])
    net.eval()
    return net, payload


def save_training_curves(out_dir: Path, rewards: list[float], durations: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(rewards, label="episode reward")
    if len(rewards) >= 100:
        moving = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        ax.plot(np.arange(99, 99 + len(moving)), moving, label="100-episode mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("DQN training reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rewards.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(durations)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Duration / steps")
    ax.set_title("Episode duration")
    fig.tight_layout()
    fig.savefig(out_dir / "durations.png", dpi=200)
    plt.close(fig)


def save_rollout_plot(path: Path, trajectory: dict[str, list[float]], rewards: list[float], cfg: dict[str, Any], title: str) -> None:
    if not trajectory.get("x"):
        return

    fig, ax = plt.subplots()
    ax.plot(trajectory["x"], trajectory["y"], "o-", markersize=2, label="agent")
    ax.plot(trajectory["x"][0], trajectory["y"][0], "s", markersize=8, label="start")
    ax.plot(trajectory["x"][-1], trajectory["y"][-1], "*", markersize=10, label="end")

    if trajectory.get("green_x") and trajectory.get("green_y"):
        ax.plot(trajectory["green_x"][-1], trajectory["green_y"][-1], "^", markersize=8, label="green")
    if trajectory.get("red_x") and trajectory.get("red_y"):
        ax.plot(trajectory["red_x"][-1], trajectory["red_y"][-1], "v", markersize=8, label="red")

    half_w = float(cfg["arena"]["width"]) / 2.0
    half_h = float(cfg["arena"]["height"]) / 2.0
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(half_h, -half_h)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}: reward={sum(rewards):.2f}, steps={len(rewards)}")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_reward_distance_plot(
    path: Path,
    trajectory: dict[str, list[float]],
    rewards: list[float],
    cfg: dict[str, Any],
    title: str,
) -> None:
    """Plot reward against distance to the green stimulus, red stimulus, and
    nearest wall as three separate panels, for a single greedy rollout.
    """
    if not trajectory.get("x") or not rewards:
        return

    half_w = float(cfg["arena"]["width"]) / 2.0
    half_h = float(cfg["arena"]["height"]) / 2.0

    xs = np.array(trajectory["x"], dtype=float)
    ys = np.array(trajectory["y"], dtype=float)
    rewards_arr = np.array(rewards, dtype=float)
    pos = np.column_stack([xs, ys])

    def component_reward(name: str) -> np.ndarray:
        vals = trajectory.get(f"{name}_reward")
        return np.array(vals, dtype=float) if vals else rewards_arr

    dist_wall = np.minimum.reduce([xs + half_w, half_w - xs, ys + half_h, half_h - ys])[: len(rewards_arr)]

    panels = [("wall", dist_wall, component_reward("wall"), "gray")]
    if trajectory.get("green_x") and trajectory.get("green_y"):
        green_xy = np.column_stack([trajectory["green_x"], trajectory["green_y"]])
        n = min(len(green_xy), len(rewards_arr))
        panels.append(("green", np.linalg.norm(pos[:n] - green_xy[:n], axis=1), component_reward("green"), "green"))
    if trajectory.get("red_x") and trajectory.get("red_y"):
        red_xy = np.column_stack([trajectory["red_x"], trajectory["red_y"]])
        n = min(len(red_xy), len(rewards_arr))
        panels.append(("red", np.linalg.norm(pos[:n] - red_xy[:n], axis=1), component_reward("red"), "red"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, dist, reward_vals, color) in zip(axes, panels):
        n = min(len(dist), len(reward_vals))
        ax.plot(dist[:n], reward_vals[:n], "o", markersize=3, color=color)
        ax.set_xlabel(f"Distance to {name}")
        ax.set_ylabel(f"{name} reward")
        ax.set_title(name)

    fig.suptitle(f"{title}: reward={sum(rewards):.2f}, steps={len(rewards)}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_rollout_gifs(
    path: Path,
    trajectory: dict[str, list[float]],
    rewards: list[float],
    cfg: dict[str, Any],
    title: str,
    fps: int = 10,
) -> None:
    """Animated counterpart to save_rollout_plot: render the agent's path
    growing step by step, with the red/green stimuli at their position each
    step, and write it out as a GIF.
    """
    if not trajectory.get("x"):
        return

    xs = trajectory["x"]
    ys = trajectory["y"]
    green_x = trajectory.get("green_x") or []
    green_y = trajectory.get("green_y") or []
    red_x = trajectory.get("red_x") or []
    red_y = trajectory.get("red_y") or []

    half_w = float(cfg["arena"]["width"]) / 2.0
    half_h = float(cfg["arena"]["height"]) / 2.0

    fig, ax = plt.subplots()
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(half_h, -half_h)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}: reward={sum(rewards):.2f}, steps={len(rewards)}")

    path_line, = ax.plot([], [], "o-", markersize=2, label="agent")
    ax.plot(xs[0], ys[0], "s", markersize=8, label="start")
    current_point, = ax.plot([], [], "*", markersize=10, label="agent (current)")
    green_point, = ax.plot([], [], "^", markersize=8, label="green")
    red_point, = ax.plot([], [], "v", markersize=8, label="red")
    ax.legend()
    fig.tight_layout()

    def update(frame: int):
        path_line.set_data(xs[: frame + 1], ys[: frame + 1])
        current_point.set_data([xs[frame]], [ys[frame]])
        if frame < len(green_x):
            green_point.set_data([green_x[frame]], [green_y[frame]])
        if frame < len(red_x):
            red_point.set_data([red_x[frame]], [red_y[frame]])
        return path_line, current_point, green_point, red_point

    anim = animation.FuncAnimation(fig, update, frames=len(xs), blit=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def save_reward_over_time_gif(
    path: Path,
    trajectory: dict[str, list[float]],
    rewards: list[float],
    title: str,
    fps: int = 10,
) -> None:
    """Animate total reward plus the green/red/wall components over time
    (one line each), growing step by step as the rollout plays out.
    """
    if not rewards:
        return

    steps = np.arange(len(rewards))
    series = [("total", np.array(rewards, dtype=float), "black")]
    for name, color in (("green", "tab:green"), ("red", "tab:red"), ("wall", "tab:gray")):
        vals = trajectory.get(f"{name}_reward")
        if vals:
            series.append((name, np.array(vals, dtype=float), color))

    fig, ax = plt.subplots()
    ax.set_xlim(0, max(len(steps) - 1, 1))
    all_vals = np.concatenate([vals for _, vals, _ in series])
    pad = 0.05 * (all_vals.max() - all_vals.min() + 1e-6)
    ax.set_ylim(all_vals.min() - pad, all_vals.max() + pad)
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title(title)

    lines = {name: ax.plot([], [], "-", label=name, color=color)[0] for name, _, color in series}
    ax.legend()
    fig.tight_layout()

    def update(frame: int):
        for name, vals, _ in series:
            lines[name].set_data(steps[: frame + 1], vals[: frame + 1])
        return list(lines.values())

    anim = animation.FuncAnimation(fig, update, frames=len(steps), blit=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def slugify(text: str) -> str:
    text = str(text)
    text = text.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")
