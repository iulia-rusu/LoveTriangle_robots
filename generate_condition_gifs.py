from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.environment import BraitenbergEnv

from dqn_common import (
    get_action_space,
    get_device,
    get_vehicle_xy,
    load_policy,
    reset_env,
    save_reward_over_time_gif,
    save_rollout_gifs,
    seed_everything,
    step_env,
    to_state_tensor,
)

CONDITIONS = ["separated", "orbit", "blocking", "crossing", "random"]


def rollout(env, policy_net, action_space, action_mode, device, rng, cfg, max_steps, random_start=True):
    state, _, _ = reset_env(
        env,
        cfg,
        rng=rng,
        random_start=random_start,
        start_margin=15.0,
        random_heading=True,
    )
    state_t = to_state_tensor(state, device)
    trajectory = {"x": [], "y": [], "red_x": [], "red_y": [], "green_x": [], "green_y": []}
    rewards: list[float] = []
    reward_components: dict[str, list[float]] = {"green": [], "red": [], "wall": []}

    for _ in range(max_steps):
        with torch.no_grad():
            action_idx = policy_net(state_t).max(1).indices.view(1, 1)
        observation, reward, terminated, _ = step_env(env, action_idx, action_space, action_mode)
        rewards.append(float(reward))
        components = getattr(env, "last_reward_components", {})
        for key in reward_components:
            reward_components[key].append(float(components.get(key, 0.0)))

        x, y = get_vehicle_xy(env)
        trajectory["x"].append(x)
        trajectory["y"].append(y)
        if getattr(env, "red_pos", None) is not None:
            trajectory["red_x"].append(float(env.red_pos[0]))
            trajectory["red_y"].append(float(env.red_pos[1]))
        if getattr(env, "green_pos", None) is not None:
            trajectory["green_x"].append(float(env.green_pos[0]))
            trajectory["green_y"].append(float(env.green_pos[1]))

        if terminated:
            break
        state_t = to_state_tensor(observation, device)

    trajectory.update({f"{key}_reward": vals for key, vals in reward_components.items()})
    return trajectory, rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a saved DQN policy on each of the 5 training conditions and save a GIF per condition.")
    parser.add_argument("--policy", type=str, default="outputs/dqn_noise/multi-separated-orbit-blocking-crossing-random/20260625_155625/policies/best_policy.pt")
    parser.add_argument("--out-dir", type=str, default="outputs/dqn_noise/multi-separated-orbit-blocking-crossing-random/20260625_155625/figures/condition_gifs")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.cpu)
    policy_net, payload = load_policy(Path(args.policy), device)
    cfg = payload["config"]

    action_mode = payload.get("action_mode", cfg.get("training", {}).get("env_action_mode", "index"))
    max_steps = int(args.steps or cfg.get("policy_run", {}).get("steps") or cfg["simulation"]["max_steps"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS:
        rng = np.random.default_rng(args.seed)
        env = BraitenbergEnv(cfg, condition=condition)
        action_space = payload.get("action_space") or get_action_space(env)
        random_start = condition != "orbit"
        trajectory, rewards = rollout(env, policy_net, action_space, action_mode, device, rng, cfg, max_steps, random_start=random_start)
        out_path = out_dir / f"{condition}.gif"
        save_rollout_gifs(out_path, trajectory, rewards, cfg, f"Greedy policy: {condition}")
        print(f"{condition}: reward={sum(rewards):.2f}, steps={len(rewards)} -> {out_path}")

        if condition == "orbit":
            # Drop the final step: it's a hit-bonus/penalty spike that dwarfs
            # the shaping reward and squashes the rest of the plot's y-range.
            trimmed_trajectory = {key: vals[:-1] for key, vals in trajectory.items()}
            trimmed_rewards = rewards[:-1]
            reward_time_path = out_dir / f"{condition}_reward_over_time.gif"
            save_reward_over_time_gif(reward_time_path, trimmed_trajectory, trimmed_rewards, f"Reward over time: {condition}")
            print(f"{condition}: reward-over-time gif -> {reward_time_path}")


if __name__ == "__main__":
    main()
