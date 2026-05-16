#!/usr/bin/env python3
"""
freedrive_viewer.py — Live camera + EE pose viewer with free-drive toggle.

Controls
--------
    f           toggle free-drive (torques off — robot is backdrivable by hand)
    s           print current EE pose to terminal
    q / Esc     quit (re-enables torque before exit)
"""

import time

import cv2
import numpy as np

from click_to_move import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CALIBRATION_DIR,
    HOME_DEG,
    MOTOR_NAMES,
    PORT,
    ROBOT_ID,
    URDF_PATH,
    SO101Kinematics,
    _joints_from_obs,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

_WIN = "SO-101  Free-Drive Viewer"


def _draw(frame: np.ndarray, ee_pos: np.ndarray, free_drive: bool) -> np.ndarray:
    h, w = frame.shape[:2]

    # EE pose (top-left)
    x, y, z = ee_pos
    cv2.putText(
        frame,
        f"EE  x={x:+.4f}  y={y:+.4f}  z={z:+.4f} m",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA,
    )

    # Free-drive badge (top-right)
    badge   = "FREE-DRIVE" if free_drive else "TORQUE ON"
    bcolour = (0, 80, 220) if free_drive else (30, 150, 30)
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bx = w - tw - 16
    cv2.rectangle(frame, (bx - 4, 6), (w - 6, 6 + th + 8), bcolour, -1)
    cv2.putText(frame, badge, (bx, 6 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Help bar (bottom)
    bar_h = 30
    frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
    cv2.putText(
        frame,
        "f = toggle free-drive    s = print pose    q / Esc = quit",
        (8, h - bar_h + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 220, 160), 1, cv2.LINE_AA,
    )

    return frame


def main() -> None:
    kin = SO101Kinematics(URDF_PATH)

    robot = SO101Follower(SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        use_degrees=True,
    ))
    robot.connect()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    else:
        print(f"[warn] Could not open camera {CAMERA_INDEX} — continuing without video")

    free_drive = False

    print("─" * 50)
    print("  SO-101 Free-Drive Viewer")
    print("  Press [f] to toggle free-drive (torques off).")
    print("  Press [q] or Esc to quit.")
    print("─" * 50)

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

    try:
        while True:
            # Grab camera frame
            ret, frame = cap.read() if cap.isOpened() else (False, None)
            if not ret:
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "No camera signal", (60, CAMERA_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 200), 2)

            # Read EE pose
            try:
                obs = robot.get_observation()
                q   = _joints_from_obs(obs)
                T   = kin.forward_kinematics(q)
                ee_pos = T[:3, 3]
            except Exception as exc:
                ee_pos = np.zeros(3)
                print(f"[warn] FK error: {exc}")

            cv2.imshow(_WIN, _draw(frame.copy(), ee_pos, free_drive))

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue

            k = key & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("f"):
                free_drive = not free_drive
                if free_drive:
                    robot.bus.disable_torque()
                    print("Free-drive ON  — robot is backdrivable")
                else:
                    robot.bus.enable_torque()
                    print("Free-drive OFF — torque restored")
            elif k == ord("s"):
                x, y, z = ee_pos
                print(f"EE  x={x:+.4f}  y={y:+.4f}  z={z:+.4f} m")

    finally:
        if free_drive:
            robot.bus.enable_torque()
            print("Torque re-enabled on exit.")
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
