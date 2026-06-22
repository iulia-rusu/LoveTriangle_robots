from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pygame

from src.config import load_config, set_seed
from src.environment import BraitenbergEnv
from src.pygame_view import PygameView


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
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    reward = None
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            _, reward, done, info = env.step(action=None)
            if done:
                print(f"Episode ended: {info['done_reason']}")
                env.reset()
                reward = None

        view.draw(env, reward)

    view.close()


if __name__ == "__main__":
    main()
