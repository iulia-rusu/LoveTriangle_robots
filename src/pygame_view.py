from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pygame

from .environment import BraitenbergEnv
from .geometry import unit_from_angle, left_axis, right_axis


class PygameView:
    def __init__(self, cfg: dict):
        pygame.init()
        self.cfg = cfg
        self.scale = int(cfg["render"]["pixels_per_unit"])
        self.width_px = int(cfg["arena"]["width"] * self.scale)
        self.height_px = int(cfg["arena"]["height"] * self.scale)
        self.panel_h = 120
        self.screen = pygame.display.set_mode((self.width_px, self.height_px + self.panel_h))
        pygame.display.set_caption("Braitenberg–RL vehicle simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.small_font = pygame.font.SysFont("Arial", 13)

    def world_to_screen(self, pos: np.ndarray) -> tuple[int, int]:
        x = int((pos[0] + self.cfg["arena"]["width"] / 2.0) * self.scale)
        y = int((self.cfg["arena"]["height"] / 2.0 - pos[1]) * self.scale)
        return x, y

    def draw(self, env: BraitenbergEnv, reward: float | None = None) -> None:
        self.screen.fill((245, 245, 245))
        pygame.draw.rect(self.screen, (40, 40, 40), (0, 0, self.width_px, self.height_px), width=5)

        self._draw_stimulus(env.red_pos, (220, 40, 40), "red")
        self._draw_stimulus(env.green_pos, (40, 170, 70), "green")
        self._draw_vehicle(env)

        if self.cfg["render"].get("show_sensor_rays", True):
            self._draw_sensor_rays(env)

        if self.cfg["render"].get("show_text", True):
            self._draw_panel(env, reward)

        pygame.display.flip()
        self.clock.tick(int(self.cfg["render"]["fps"]))

    def _draw_stimulus(self, pos: Optional[np.ndarray], colour: tuple[int, int, int], label: str) -> None:
        if pos is None:
            return
        xy = self.world_to_screen(pos)
        pygame.draw.circle(self.screen, colour, xy, 13)
        txt = self.small_font.render(label, True, colour)
        self.screen.blit(txt, (xy[0] + 12, xy[1] - 10))

    def _draw_vehicle(self, env: BraitenbergEnv) -> None:
        st = env.vehicle.state
        pos = st.position
        centre = self.world_to_screen(pos)
        radius = int(float(self.cfg["vehicle"]["radius"]) * self.scale)
        radius = max(radius, 12)
        pygame.draw.circle(self.screen, (40, 110, 220), centre, radius)

        heading_vec = unit_from_angle(st.heading)
        front = self.world_to_screen(pos + heading_vec * 0.28)
        rear = self.world_to_screen(pos - heading_vec * 0.18)
        pygame.draw.circle(self.screen, (0, 50, 200), front, 6)
        pygame.draw.circle(self.screen, (240, 200, 40), rear, 5)
        pygame.draw.line(self.screen, (20, 20, 20), centre, front, width=2)

    def _draw_sensor_rays(self, env: BraitenbergEnv) -> None:
        st = env.vehicle.state
        origin = self.world_to_screen(st.position)
        

    def _draw_panel(self, env: BraitenbergEnv, reward: float | None) -> None:
        y0 = self.height_px + 8
        info = env.last_info

        lines = [
            f"step={env.step_idx}  t={env.time:.1f}  "
            f"condition={env.condition}  reward={0 if reward is None else reward:.3f}",
        ]

        if info is not None:
            lines.append(
                f"left_motor={info.left_motor:.3f}  "
                f"right_motor={info.right_motor:.3f}"
            )

        for i, line in enumerate(lines):
            text = self.font.render(line, True, (20, 20, 20))
            self.screen.blit(text, (10, y0 + i * 24))

    def close(self) -> None:
        pygame.quit()

