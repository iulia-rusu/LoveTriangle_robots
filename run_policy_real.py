"""
Run a hand-designed navigation policy on the real LBP robot (or in simulation).

State space: flat index over (x, y, orientation), x in [0, N_COLS), y in
[0, N_ROWS), orientation in {N, E, S, W}. Origin (0, 0) is the arena's top-left
corner; y increases downward.

Action space (heading-relative): stay, forward, turn_left, turn_right.

Position tracking (simulation=False): Bonsai publishes the robot's two tracked
markers (red = front, green = back) over OSC as /robot [x_red, y_red, x_green,
y_green] in raw camera pixel coordinates on port 9000. These get mapped into
the state-space coordinate frame via a perspective transform calibrated from
the arena's four corners in pixel space (ARENA_CORNERS_PX below).

Requires:  pip install opencv-python numpy
"""

from __future__ import annotations

import sys
import os
import time
import threading

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from robot_client import RobotClient, OscListener

# ── State space ───────────────────────────────────────────────────────────────

CELL_SIZE_CM = 1  # real-world cm per grid cell

height = 240
width = 180

N_COLS = width   # discrete x states, x in [0, width)
N_ROWS = height  # discrete y states, y in [0, height)

ORIENTATIONS = {0: "N", 1: "E", 2: "S", 3: "W"}
ORIENTATION_TO_IDX = {name: idx for idx, name in ORIENTATIONS.items()}
N_ORIENTATIONS = len(ORIENTATIONS)


def in_bounds(x, y):
    return 0 <= x < N_COLS and 0 <= y < N_ROWS


def get_position(state):
    """Convert a flat state index into (x, y, orientation).
    State indices run row-major over (x, y), with orientation as the
    fastest-varying component: state = (y * N_COLS + x) * N_ORIENTATIONS + orientation_idx
    """
    total_states = N_COLS * N_ROWS * N_ORIENTATIONS
    if not (0 <= state < total_states):
        raise ValueError(f"state {state} is out of bounds for a {N_COLS}x{N_ROWS}x{N_ORIENTATIONS} state space")
    o_idx = state % N_ORIENTATIONS
    grid_idx = state // N_ORIENTATIONS
    x = grid_idx % N_COLS
    y = grid_idx // N_COLS
    return x, y, ORIENTATIONS[o_idx]


def get_state(x, y, orientation):
    """orientation: 'N'/'E'/'S'/'W' or its index 0-3."""
    o_idx = orientation if isinstance(orientation, int) else ORIENTATION_TO_IDX[orientation]
    return (y * N_COLS + x) * N_ORIENTATIONS + o_idx


# ── Action space ──────────────────────────────────────────────────────────────

ACTIONS = {0: "stay", 1: "forward", 2: "turn_left", 3: "turn_right"}

MOTOR_STEP_CM = 5  # measured real-world displacement of one 'forward' action
STEP_SIZE = max(1, round(MOTOR_STEP_CM / CELL_SIZE_CM))  # displacement in grid cells

# forward displacement by current orientation index (N, E, S, W)
FORWARD_DELTA = {0: (0, -STEP_SIZE), 1: (STEP_SIZE, 0), 2: (0, STEP_SIZE), 3: (-STEP_SIZE, 0)}


def transition(state, action):
    """RL transition function T(state, action) -> next_state, heading-aware."""
    x, y, orientation = get_position(state)
    o_idx = ORIENTATION_TO_IDX[orientation]

    if action == 1:  # forward
        dx, dy = FORWARD_DELTA[o_idx]
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny):
            x, y = nx, ny
    elif action == 2:  # turn_left
        o_idx = (o_idx - 1) % 4
    elif action == 3:  # turn_right
        o_idx = (o_idx + 1) % 4
    # action == 0 (stay): no change

    return get_state(x, y, o_idx)


def action_to_motor(action):
    """Convert a heading-relative action into a motor command for the robot.
    Returns (left_wheel, right_wheel, duration).
    """
    if action == 0:  # stay
        return np.array([0, 0, 1])
    elif action == 1:  # forward
        return np.array([30, 30, 0.5])
    elif action == 2:  # turn_left
        return np.array([0, 44, 1])
    elif action == 3:  # turn_right
        return np.array([44, 0, 1])
    else:
        raise ValueError(f"Unknown action {action}")


# ── Goal region + hand-designed policy ────────────────────────────────────────

def states_in_region(x_range, y_range, orientations=None):
    """Return all flat state indices whose (x, y) position falls in the given ranges
    and whose orientation is in `orientations` (default: any orientation).
    """
    allowed = set(ORIENTATIONS.values()) if orientations is None else {
        o if isinstance(o, str) else ORIENTATIONS[o] for o in orientations
    }

    total_states = N_COLS * N_ROWS * N_ORIENTATIONS
    states = []
    for state in range(total_states):
        x, y, o = get_position(state)
        if x_range[0] <= x < x_range[1] and y_range[0] <= y < y_range[1] and o in allowed:
            states.append(state)
    return states


def desired_orientation_idx(dx, dy):
    """Pick the cardinal heading that best reduces the larger remaining offset."""
    if abs(dx) >= abs(dy) and dx != 0:
        return 1 if dx > 0 else 3  # E : W
    elif dy != 0:
        return 2 if dy > 0 else 0  # S : N
    return None  # already at goal


def design_stationary_policy(goal_states):
    """Deterministic tabular policy: state -> action, that turns the agent to
    face the nearest goal cell, then drives forward. We assume the goal region
    is stationary, so the policy depends only on the agent's own state.
    """
    goal_xy = {get_position(s)[:2] for s in goal_states}
    total_states = N_COLS * N_ROWS * N_ORIENTATIONS
    policy = {}

    for state in range(total_states):
        x, y, orientation = get_position(state)
        o_idx = ORIENTATION_TO_IDX[orientation]

        if (x, y) in goal_xy:
            policy[state] = 0  # stay, already in goal region
            continue

        gx, gy = min(goal_xy, key=lambda g: abs(g[0] - x) + abs(g[1] - y))
        target_idx = desired_orientation_idx(gx - x, gy - y)

        if target_idx == o_idx:
            action = 1  # forward
        else:
            diff = (target_idx - o_idx) % 4
            action = 3 if diff in (1, 2) else 2  # turn_right (incl. 180 tie-break) : turn_left

        policy[state] = action

    return policy


def reward(state, goal_states, step_cost=-1, goal_reward=10):
    """Reward for being in `state`. Depends only on reaching the goal region:
    a fixed cost per timestep, plus a bonus once inside the goal region.
    """
    x, y, _ = get_position(state)
    goal_xy = {get_position(s)[:2] for s in goal_states}
    return goal_reward if (x, y) in goal_xy else step_cost


# ── Camera pixel <-> state-space coordinate mapping ───────────────────────────

# arena corners in raw camera pixel coordinates (calibrated once, camera isn't axis-aligned with the arena)
ARENA_CORNERS_PX = np.array([
    [1217, 400],  # top left
    [157, 890],   # top right
    [1159, 71],   # bottom left
    [179, 172],   # bottom right
], dtype=np.float32)

# corresponding corners in our state-space coordinate frame
ARENA_CORNERS_STATE = np.array([
    [0, 0],
    [N_COLS - 1, 0],
    [0, N_ROWS - 1],
    [N_COLS - 1, N_ROWS - 1],
], dtype=np.float32)

_ARENA_HOMOGRAPHY = cv2.getPerspectiveTransform(ARENA_CORNERS_PX, ARENA_CORNERS_STATE)
_ARENA_HOMOGRAPHY_INV = cv2.getPerspectiveTransform(ARENA_CORNERS_STATE, ARENA_CORNERS_PX)


def pixel_to_arena(px, py):
    """Map a raw camera pixel coordinate to (x, y) in the state-space coordinate frame."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, _ARENA_HOMOGRAPHY)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def arena_to_pixel(x, y):
    """Map a state-space (x, y) coordinate back to raw camera pixel coordinates
    (the inverse of pixel_to_arena), e.g. for overlaying the goal region on the
    live camera frame.
    """
    pt = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, _ARENA_HOMOGRAPHY_INV)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def goal_states_to_pixels(goal_states):
    """Project each goal state's (x, y) cell into camera pixel coordinates."""
    return [arena_to_pixel(x, y) for x, y in {get_position(s)[:2] for s in goal_states}]


def vector_to_orientation(dx, dy):
    """Snap a direction vector to the nearest cardinal orientation used by our state space."""
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


def has_invalid_marker(robot_data):
    """Bonsai sends -1 for any marker (red/green) it couldn't detect in the current frame."""
    return any(v == -1 for v in robot_data)


def robot_to_state(robot_data):
    """Convert the robot's two-marker camera readout into a discrete state.
    robot_data: [x_red, y_red, x_green, y_green] in raw camera pixel coordinates.
    Red is mounted on the front of the robot, green on the back; the vector
    from green -> red gives its heading.
    """
    x_red_px, y_red_px, x_green_px, y_green_px = robot_data

    x_red, y_red = pixel_to_arena(x_red_px, y_red_px)
    x_green, y_green = pixel_to_arena(x_green_px, y_green_px)

    orientation = vector_to_orientation(x_red - x_green, y_red - y_green)

    x = min(max(round(x_red), 0), N_COLS - 1)
    y = min(max(round(y_red), 0), N_ROWS - 1)
    return get_state(x, y, orientation)


def wait_for_camera_frame(robot, timeout=5.0, poll_interval=0.1):
    """Block until robot.frame has a first image, or until timeout. Returns
    True if a frame arrived, False on timeout.
    """
    waited = 0.0
    while robot.frame is None and waited < timeout:
        time.sleep(poll_interval)
        waited += poll_interval
    return robot.frame is not None


def show_preview(robot, robot_data, goal_states, window_name="Preview"):
    """Overlay the tracked red/green markers and the goal region on the live
    camera frame, so you can visually confirm the robot is in frame and being
    tracked correctly before driving it.
    """
    frame = robot.frame
    if frame is None:
        return
    vis = frame.copy()
    for gx, gy in goal_states_to_pixels(goal_states):
        cv2.circle(vis, (int(gx), int(gy)), 2, (0, 255, 0), -1)
    if has_invalid_marker(robot_data):
        cv2.putText(vis, "marker not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        x_red, y_red, x_green, y_green = robot_data
        cv2.circle(vis, (int(x_red), int(y_red)), 6, (0, 0, 255), -1)
        cv2.circle(vis, (int(x_green), int(y_green)), 6, (0, 200, 0), -1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vis)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)


# ── Rollout ────────────────────────────────────────────────────────────────────

def run_policy(start_state, policy, goal_states, max_steps=500, simulation=False, position_timeout=5.0, settle_time=0.2):
    """Roll out `policy` from `start_state` until it reaches the goal region or max_steps.
    When simulation=False, the next state (and the initial state) is read from
    Bonsai's /robot OSC feed (port 9000) instead of the hand-coded transition()
    model and the passed-in start_state — we don't actually know where the robot
    is until we hear from Bonsai, so start_state is only used in simulation mode.

    On real hardware:
    - Before driving, shows a live preview window with the tracked markers and
      goal region overlaid, and blocks on an Enter-to-start prompt so you can
      confirm tracking looks right first.
    - Each step: drive for `duration`, stop the wheels, wait `settle_time` for
      the tracker to catch up, then read the now-stationary position. If a
      marker read comes back invalid (-1), the state update for that step is
      skipped and the previous state is kept instead.
    """
    goal_xy = {get_position(s)[:2] for s in goal_states}

    robot = RobotClient()
    robot.start()

    robot_data = [0, 0, 0, 0]  # [x_red, y_red, x_green, y_green] (raw camera pixels)
    got_position = False

    def on_robot(args):
        nonlocal got_position
        robot_data[:] = args
        got_position = True

    positions = None
    if not simulation:
        positions = OscListener(port=9000)
        positions.subscribe("/robot", on_robot)
        positions.start()

        waited = 0.0
        while not got_position and waited < position_timeout:
            time.sleep(0.1)
            waited += 0.1
        if not got_position:
            robot.stop()
            raise RuntimeError(f"Timed out after {position_timeout}s waiting for initial /robot position from Bonsai")

        print("Waiting for camera frame...")
        if not wait_for_camera_frame(robot, timeout=position_timeout):
            robot.stop()
            raise RuntimeError(f"Timed out after {position_timeout}s waiting for a camera frame from the Pi")

        preview_window = "Preview - confirm tracking, then check console"
        print("Showing preview window — press Enter in this console to start.")
        stop_preview = threading.Event()

        def _refresh_preview():
            while not stop_preview.is_set():
                show_preview(robot, robot_data, goal_states, preview_window)
                cv2.waitKey(30)

        preview_thread = threading.Thread(target=_refresh_preview, daemon=True)
        preview_thread.start()
        input("Press Enter to start driving the robot...")
        stop_preview.set()
        preview_thread.join(timeout=1.0)
        cv2.destroyWindow(preview_window)

        if has_invalid_marker(robot_data):
            robot.stop()
            raise RuntimeError("Marker not detected at start — check camera/tracking before running.")
        state = robot_to_state(robot_data)
    else:
        state = start_state

    history = [get_position(state)[:2]]
    rewards = []

    for _ in range(max_steps):
        r = reward(state, goal_states)
        rewards.append(r)
        x, y, _ = get_position(state)
        if (x, y) in goal_xy:
            break
        action = policy[state]
        left, right, duration = action_to_motor(action)
        robot.set_wheels(left, right)  # drive
        print(f"State: {state}, Position: ({x}, {y}), Action: {ACTIONS[action]}, Wheels: (L={left}, R={right}), Duration: {duration}s")
        time.sleep(duration)
        if simulation:
            state = transition(state, action)
        else:
            robot.set_wheels(0, 0)   # stop moving before reading position
            time.sleep(settle_time)  # let the tracker catch up to the now-stationary robot
            if has_invalid_marker(robot_data):
                print("Warning: marker not detected this step, keeping previous state.")
            else:
                state = robot_to_state(robot_data)
        history.append(get_position(state)[:2])

    print("Goal reached!" if rewards[-1] > 0 else "Max steps reached without reaching goal.")
    robot.set_wheels(0, 0)
    robot.stop()
    if positions is not None:
        positions.stop()
    return history, rewards


def main():
    # upper center: top 1/6 of the arena (in y), middle 1/4 of the width (in x), any orientation
    upper_band = N_ROWS // 6
    center_width = N_COLS // 4
    x_start = (N_COLS - center_width) // 2
    x_end = x_start + center_width
    goal_states = states_in_region(x_range=(x_start, x_end), y_range=(0, upper_band))

    policy = design_stationary_policy(goal_states)
    start_state = get_state(x=160, y=200, orientation="N")

    history, rewards = run_policy(start_state, policy, goal_states, simulation=False)

    print(f"Steps: {len(rewards)}, total reward: {sum(rewards)}, final position: {history[-1]}")


if __name__ == "__main__":
    main()
