from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.environment import BraitenbergEnv
from src.pygame_view import PygameView
from src.DQN import apply_overrides, normalise_step_result, reset_env, resolve_condition, seed_everything


def hard_exit(*args):
    print("Hard exit", flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, hard_exit)
signal.signal(signal.SIGTERM, hard_exit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the environment with the built-in/no-action controller and Pygame view.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--start-margin", type=float, default=0.0)
    parser.add_argument("--fixed-heading", action="store_true")
    parser.add_argument("--pixels-per-unit", type=float, default=None, help="Override render.pixels_per_unit. Useful for large arenas.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    if args.pixels_per_unit is not None:
        cfg.setdefault("render", {})["pixels_per_unit"] = float(args.pixels_per_unit)
    condition = resolve_condition(cfg, args.condition)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    if callable(set_seed):
        set_seed(seed)
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    env = BraitenbergEnv(cfg, condition=condition)
    reset_env(
        env,
        cfg,
        rng=rng,
        random_start=args.random_start,
        start_margin=args.start_margin,
        random_heading=not args.fixed_heading,
    )
    view = PygameView(cfg)

    paused = False
    reward = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                hard_exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    hard_exit()
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    reset_env(
                        env,
                        cfg,
                        rng=rng,
                        random_start=args.random_start,
                        start_margin=args.start_margin,
                        random_heading=not args.fixed_heading,
                    )
                    reward = None

        if not paused:
            observation, reward, done, info = normalise_step_result(env.step(action=None))
            if done:
                reason = info.get("done_reason") or getattr(getattr(env, "last_info", None), "done_reason", "unknown")
                print(f"Episode ended: {reason}", flush=True)
                reset_env(
                    env,
                    cfg,
                    rng=rng,
                    random_start=args.random_start,
                    start_margin=args.start_margin,
                    random_heading=not args.fixed_heading,
                )
                reward = None

        view.draw(env, reward)


if __name__ == "__main__":
    main()
