from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pygame

from src.config import load_config, set_seed
from src.environment import BraitenbergEnv
from src.pygame_view import PygameView


def hard_exit(*args):
    print("Hard exit", flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, hard_exit)
signal.signal(signal.SIGTERM, hard_exit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--condition", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))

    env = BraitenbergEnv(cfg, condition=args.condition)
    env.reset()
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
                    env.reset()
                    reward = None

        if not paused:
            state, reward, done = env.step(action=None)

            if done:
                print(f"Episode ended: {info.done_reason}", flush=True)
                env.reset()
                reward = None

        view.draw(env, reward)


if __name__ == "__main__":
    main()