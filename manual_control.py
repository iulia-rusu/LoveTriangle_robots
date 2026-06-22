"""
Manual keyboard teleoperation for the LBP robot.

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
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from robot_client import RobotClient

try:
    from pynput import keyboard
except ImportError:
    sys.exit("Install pynput:  pip install pynput")

SPEED = 50

COMMANDS = {
    "w":     ( SPEED,  SPEED),   # forward
    "s":     (-SPEED, -SPEED),   # backward
    "a":     (-SPEED,  SPEED),   # spin left
    "d":     ( SPEED, -SPEED),   # spin right
    " ":     (0,       0),       # stop
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


def _update(robot: RobotClient):
    """Apply the highest-priority held key, or stop if none."""
    for k in ("w", "s", "a", "d", " "):
        if k in _HELD:
            vl, vr = COMMANDS[k]
            robot.set_wheels(vl, vr)
            print(f"\r  [{LABELS[k]:8s}]  wheels=({vl:+d}, {vr:+d})  "
                  f"bumpers={robot.bumpers}  analogs={robot.analogs}    ",
                  end="", flush=True)
            return
    robot.stop_wheels()
    print(f"\r  [stop    ]  wheels=(  0,   0)  "
          f"bumpers={robot.bumpers}  analogs={robot.analogs}    ",
          end="", flush=True)


def run():
    print("LBP Manual Control — W/A/S/D to move, Space to stop, Esc to quit\n")

    try:
        import cv2
        have_cv2 = True
    except ImportError:
        print("[warning] opencv-python not installed — camera window disabled.")
        have_cv2 = False

    with RobotClient() as robot:
        stop = threading.Event()

        def on_press(key):
            ch = _key_char(key)
            if ch in COMMANDS:
                _HELD.add(ch)
                _update(robot)
            elif key == keyboard.Key.esc:
                stop.set()
                return False  # stop listener

        def on_release(key):
            ch = _key_char(key)
            if ch in _HELD:
                _HELD.discard(ch)
                _update(robot)

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            if have_cv2:
                cv2.namedWindow("LBP Camera", cv2.WINDOW_NORMAL)
                while not stop.is_set():
                    frame = robot.frame
                    if frame is not None:
                        cv2.imshow("LBP Camera", frame)
                    if cv2.waitKey(30) == 27:  # Esc pressed in cv2 window
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
