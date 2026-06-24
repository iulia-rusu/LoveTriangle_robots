from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pygame
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.environment import BraitenbergEnv
from src.pygame_view import PygameView

from dqn_common import (
    apply_overrides,
    get_action_space,
    get_device,
    get_vehicle_xy,
    load_policy,
    reset_env,
    resolve_condition,
    save_rollout_plot,
    seed_everything,
    step_env,
    to_state_tensor,
)


def choose(cli_value: Any, cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else cfg.get(section, {}).get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a saved DQN policy in BraitenbergEnv.")
    parser.add_argument("--policy", type=str, default="outputs/dqn/policies/best_policy.pt")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to config saved in the policy.")
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--start-margin", type=float, default=None)
    parser.add_argument("--random-heading", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pixels-per-unit", type=float, default=None, help="Override render.pixels_per_unit for visualisation.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--env-action-mode", choices=["index", "motor"], default=None, help="Override action mode saved in the policy.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy)

    # Loading uses CPU/GPU choice, but CPU is usually simplest for rollouts.
    seed = int(args.seed if args.seed is not None else 0)
    seed_everything(seed)

    # Temporarily load on CPU if the config has not yet been read.
    tmp_device = get_device(bool(args.cpu) if args.cpu is not None else False)
    policy_net, payload = load_policy(policy_path, tmp_device)

    cfg = load_config(args.config) if args.config else payload["config"]
    cfg = apply_overrides(cfg, args.overrides)

    use_cpu = bool(choose(args.cpu, cfg, "policy_run", "cpu", False))
    device = get_device(use_cpu)
    if device != tmp_device:
        policy_net, payload = load_policy(policy_path, device)

    condition = resolve_condition(cfg, args.condition)
    if args.pixels_per_unit is not None:
        cfg.setdefault("render", {})["pixels_per_unit"] = float(args.pixels_per_unit)
    else:
        ppu = cfg.get("policy_run", {}).get("pixels_per_unit", None)
        if ppu is not None:
            cfg.setdefault("render", {})["pixels_per_unit"] = float(ppu)

    random_start = bool(choose(args.random_start, cfg, "policy_run", "random_start", False))
    start_margin = float(choose(args.start_margin, cfg, "policy_run", "start_margin", cfg.get("training", {}).get("start_margin", 0.0)))
    random_heading = bool(choose(args.random_heading, cfg, "policy_run", "random_heading", cfg.get("training", {}).get("random_heading", True)))
    render = bool(choose(args.render, cfg, "policy_run", "render", False))
    out = str(choose(args.out, cfg, "policy_run", "out", "outputs/dqn/figures/policy_rollout.png"))

    steps_cfg = cfg.get("policy_run", {}).get("steps", None)
    max_steps = int(args.steps or steps_cfg or cfg["simulation"]["max_steps"])

    action_mode = args.env_action_mode or payload.get("action_mode", cfg.get("training", {}).get("env_action_mode", "index"))
    rng = np.random.default_rng(seed)

    env = BraitenbergEnv(cfg, condition=condition)
    state, _, start_pose = reset_env(
        env,
        cfg,
        rng=rng,
        random_start=random_start,
        start_margin=start_margin,
        random_heading=random_heading,
    )
    state_t = to_state_tensor(state, device)
    action_space = payload.get("action_space") or get_action_space(env)

    view = PygameView(cfg) if render else None
    trajectory = {"x": [], "y": [], "red_x": [], "red_y": [], "green_x": [], "green_y": []}
    rewards: list[float] = []

    for _ in range(max_steps):
        if view is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    view.close()
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    view.close()
                    return

        with torch.no_grad():
            action_idx = policy_net(state_t).max(1).indices.view(1, 1)

        observation, reward, terminated, _ = step_env(env, action_idx, action_space, action_mode)
        rewards.append(float(reward))

        x, y = get_vehicle_xy(env)
        trajectory["x"].append(x)
        trajectory["y"].append(y)
        if getattr(env, "red_pos", None) is not None:
            trajectory["red_x"].append(float(env.red_pos[0]))
            trajectory["red_y"].append(float(env.red_pos[1]))
        if getattr(env, "green_pos", None) is not None:
            trajectory["green_x"].append(float(env.green_pos[0]))
            trajectory["green_y"].append(float(env.green_pos[1]))

        if view is not None:
            view.draw(env, reward)

        if terminated:
            break
        state_t = to_state_tensor(observation, device)

    if view is not None:
        view.close()

    if not args.no_plot:
        save_rollout_plot(Path(out), trajectory, rewards, cfg, "Greedy policy rollout")
        print(f"Saved rollout plot to {out}")

    print(
        f"reward={sum(rewards):.3f}; steps={len(rewards)}; "
        f"start=({start_pose.x:.3f}, {start_pose.y:.3f}, {start_pose.heading:.3f})"
    )


if __name__ == "__main__":
    main()
