from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Per-seed state for the "random" condition: each stimulus is a 2D point with
# a velocity that gets nudged by random acceleration ("jitter") and damped
# every call, then integrated forward by the elapsed time since the last
# call. Keyed by seed so independent envs/episodes using different seeds
# don't trample each other's walk.
_RANDOM_WALK_STATE: dict[int, dict] = {}

_WALK_DAMPING = 0.95      # velocity decay per second, keeps the walk from drifting off forever
_WALK_JITTER_STD = 40.0   # std of the random acceleration applied each second
_WALK_SPEED_LIMIT = 50.0  # cap on velocity magnitude
_WALK_BOUND = 40.0        # half-range each stimulus starts within at the beginning of a walk


def _new_random_walk_state(rng: np.random.Generator) -> dict:
    return {
        "rng": rng,
        "red_pos": rng.uniform(-_WALK_BOUND, _WALK_BOUND, size=2).astype(np.float32),
        "red_vel": np.zeros(2, dtype=np.float32),
        "green_pos": rng.uniform(-_WALK_BOUND, _WALK_BOUND, size=2).astype(np.float32),
        "green_vel": np.zeros(2, dtype=np.float32),
        "last_t": 0.0,
    }


def _advance_random_walk(state: dict, dt: float) -> None:
    rng = state["rng"]
    for name in ("red", "green"):
        vel = state[f"{name}_vel"] * _WALK_DAMPING + rng.normal(0.0, _WALK_JITTER_STD, size=2) * dt
        speed = float(np.linalg.norm(vel))
        if speed > _WALK_SPEED_LIMIT:
            vel = vel * (_WALK_SPEED_LIMIT / speed)
        state[f"{name}_vel"] = vel.astype(np.float32)
        state[f"{name}_pos"] = (state[f"{name}_pos"] + vel * dt).astype(np.float32)


def stimulus_positions(
    condition: str, t: float, seed: int | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return red and green stimulus positions for a named condition.

    seed: only used by condition="random", to pick which independent random
    walk to follow (see _RANDOM_WALK_STATE). Different seeds walk
    independently; the same seed continues the same walk across calls.
    Defaults to 0 if not given. t == 0.0 (e.g. from env.reset()) starts a
    fresh walk for that seed; any later t advances the existing walk by the
    elapsed time since the previous call.
    """
    if condition == "red_only":
        return np.array([56.0, 0.0], dtype=np.float32), None

    if condition == "green_only":
        return None, np.array([12.0, 0.0], dtype=np.float32)

    if condition == "separated":
        red = np.array([50, 20], dtype=np.float32)
        green = np.array([-80, -40], dtype=np.float32)
        return red, green

    if condition == "blocking":
        green = np.array([56.0, 0.0], dtype=np.float32)

        red = np.array([14.0, 24.0 * math.sin(0.7 * t)], dtype=np.float32)
        return red, green

    if condition == "crossing":
        red = np.array([56.0 * math.cos(0.25 * t), 36.0 * math.sin(0.25 * t)], dtype=np.float32)
        green = np.array([48.0 * math.cos(0.25 * t + math.pi), 28.0 * math.sin(0.25 * t + math.pi)], dtype=np.float32)
        return red, green

    if condition == "orbit":
        red = np.array([60.0 * math.cos(0.35 * t), 70.0 * math.sin(0.35 * t)], dtype=np.float32)
        green = np.array([35.0 * math.cos(0.65 * t + 1.5), 35.0 * math.sin(0.65 * t + 1.5)], dtype=np.float32)
        return red, green

    if condition == "random":
        walk_seed = seed if seed is not None else 0
        state = _RANDOM_WALK_STATE.get(walk_seed)
        if state is None or t == 0.0:
            state = _new_random_walk_state(np.random.default_rng(walk_seed))
            _RANDOM_WALK_STATE[walk_seed] = state
        else:
            dt = t - state["last_t"]
            if dt > 0:
                _advance_random_walk(state, dt)
        state["last_t"] = t
        return state["red_pos"].copy(), state["green_pos"].copy()

    raise ValueError(f"Unknown condition: {condition}")
