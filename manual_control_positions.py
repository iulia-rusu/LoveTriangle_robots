"""
Manual keyboard teleoperation for the LBP robot — with position display.

Extends manual_control with a live readout of /robot and /auxrobots
received from Bonsai on a separate OSC listener (port 9001).

Keys:
  W     — forward
  S     — backward
  A     — spin left
  D     — spin right
  Space — stop
  Esc   — quit

Requires:  pip install pynput
"""

from __future__ import annotations

import threading
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from robot_client import RobotClient, OscListener

try:
    from pynput import keyboard
except ImportError:
    sys.exit("Install pynput:  pip install pynput")

SPEED = 50

COMMANDS = {
    "w":     ( SPEED,  SPEED),
    "s":     (-SPEED, -SPEED),
    "a":     (-SPEED,  SPEED),
    "d":     ( SPEED, -SPEED),
    " ":     (0,       0),
}

LABELS = {
    "w": "forward", "s": "backward",
    "a": "left",    "d": "right", " ": "stop",
}

_HELD: set = set()


def _key_char(key) -> str | None:
    try:
        return key.char.lower() if key.char else None
    except AttributeError:
        if key == keyboard.Key.space:
            return " "
        return None


def _update(robot: RobotClient, robot_data: list, aux_robots: list):
    """Apply the highest-priority held key, or stop if none."""
    for k in ("w", "s", "a", "d", " "):
        if k in _HELD:
            vl, vr = COMMANDS[k]
            robot.set_wheels(vl, vr)
            _print_status(LABELS[k], vl, vr, robot, robot_data, aux_robots)
            return
    robot.stop_wheels()
    _print_status("stop", 0, 0, robot, robot_data, aux_robots)


def _print_status(label: str, vl: int, vr: int,
                  robot: RobotClient, robot_data: list, aux_robots: list):
    print(
        f"\r  [{label:8s}]  "
        f"wheels=({vl:+4d},{vr:+4d})  "
        f"bumpers={robot.bumpers}  "
        f"analogs={robot.analogs}  "
        f"robot={robot_data}  "
        f"aux={aux_robots}    ",
        end="", flush=True,
    )


def run():
    print("LBP Manual Control (positions) — W/A/S/D to move, Space to stop, Esc to quit\n")
    print("  Robot OSC on port 2390 | Position OSC listener on port 9000\n")

    try:
        import cv2
        have_cv2 = True
    except ImportError:
        print("[warning] opencv-python not installed — camera window disabled.")
        have_cv2 = False

    robot_data = [0, 0, 0, 0]
    aux_robots  = [0, 0, 0, 0]

    with RobotClient() as robot, OscListener(port=9000) as positions:
        stop = threading.Event()

        def on_robot(args):
            robot_data[:] = args
            _update(robot, robot_data, aux_robots)

        def on_aux_robots(args):
            aux_robots[:] = args
            _update(robot, robot_data, aux_robots)

        positions.subscribe("/robot",     on_robot)
        positions.subscribe("/auxrobots", on_aux_robots)

        def on_press(key):
            ch = _key_char(key)
            if ch in COMMANDS:
                _HELD.add(ch)
                _update(robot, robot_data, aux_robots)
            elif key == keyboard.Key.esc:
                stop.set()
                return False

        def on_release(key):
            ch = _key_char(key)
            if ch in _HELD:
                _HELD.discard(ch)
                _update(robot, robot_data, aux_robots)

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            if have_cv2:
                cv2.namedWindow("LBP Camera", cv2.WINDOW_NORMAL)
                while not stop.is_set():
                    frame = robot.frame
                    if frame is not None:
                        cv2.imshow("LBP Camera", frame)
                    if cv2.waitKey(30) == 27:
                        stop.set()
                        break
                cv2.destroyAllWindows()
                listener.stop()
            else:
                listener.join()

        robot.stop_wheels()
        print("\nDisconnected.")


if __name__ == "__main__":
    run()
