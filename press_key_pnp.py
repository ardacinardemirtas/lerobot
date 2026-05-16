#!/usr/bin/env python3
"""
press_key_pnp.py — PnP-based two-step keyboard key pressing for SO-101.

Replaces the ray-plane intersection in press_key.py with a PnP solve over
all visible keys, giving a true 3D position for each key without any
hardcoded surface-Z assumption.

Pipeline (per press)
--------------------
Step 1  (Coarse / hover)
    Detect all visible keys → PnP → get target key's 3D base-frame position.
    Move EE to (x_key, y_key, z_key + HOVER_OFFSET_M).

Step 2  (Fine / press)
    From the closer vantage, detect again → PnP → refined position.
    Move EE to (x_key, y_key, z_key - PRESS_BELOW_M)  — slightly below the
    key-top surface so the arm's compliance depresses the key.
    Lift back to hover height.

Usage
-----
    python press_key_pnp.py --key a
    python press_key_pnp.py --key enter --no-lift

Requires
--------
    Roboflow inference server running  (see keyboard_detection/start_inference.sh)
    ROBOFLOW_API_KEY in env or keyboard_detection/.env.inference
    Robot connected on PORT (see click_to_move.py CONFIG)
"""

import argparse
import base64
import os
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

# ── Load .env.inference ───────────────────────────────────────────────────────
_ENV_FILE = Path(__file__).parent / "keyboard_detection" / ".env.inference"
if _ENV_FILE.exists():
    for _ln in _ENV_FILE.read_text().splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from click_to_move import (  # noqa: E402
    CAMERA_K,
    DIST_COEFFS,
    T_EE_CAM,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    PORT,
    ROBOT_ID,
    CALIBRATION_DIR,
    URDF_PATH,
    CAMERA_INDEX,
    HOME_DEG,
    WS_MIN,
    WS_MAX,
    MOTOR_NAMES,
    FPS,
)
from move_to_position_qp import (  # noqa: E402
    SO101Kinematics,
    smooth_move,
    _joints_from_obs,
    SETTLE_THRESHOLD_M,
    FIXED_WRIST_ROLL_DEG,
    JOINT_INDEX,
)
from keyboard_pnp import (  # noqa: E402
    detections_from_roboflow,
    get_key_positions_in_base_frame,
    locate_key_in_base_frame,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Metres above the PnP-derived key-top surface to hover before the fine step.
HOVER_OFFSET_M = 0.04

# How far below the PnP-derived key-top surface to aim for the press.
# The arm physically can't reach this — stall detection stops it at contact.
# Larger value = more assertive descent before stall fires.
PRESS_BELOW_M = 0.008

# Press-settle stall detection (ported from move_to_position.py)
# If the EE moves less than STALL_MIN_M over STALL_WINDOW consecutive steps
# it has made contact with the key and we stop immediately.
_STALL_WINDOW   = 10     # ~0.33 s at 30 Hz
_STALL_MIN_M    = 0.0004 # < 0.4 mm travel → contact
_PRESS_MAX_ITER = 60     # hard cap (~2 s)

# Roboflow inference
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST",    "http://localhost:9001")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY",  "")
MODEL_ID         = os.environ.get("ROBOFLOW_MODEL_ID", "keyboard-key-recognition-kw7nc/14")
CONF_THRESHOLD   = 0.50

# Minimum PnP inliers required to trust a pose estimate
MIN_PNP_INLIERS = 6

# ─── Detection ────────────────────────────────────────────────────────────────

def _encode_frame(frame: np.ndarray) -> str:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def detect_keys(frame: np.ndarray) -> list[dict]:
    """POST frame to Roboflow, return prediction list."""
    resp = requests.post(
        f"{INFERENCE_HOST.rstrip('/')}/infer/object_detection",
        json={
            "id":         str(uuid.uuid4()),
            "model_id":   MODEL_ID,
            "api_key":    ROBOFLOW_API_KEY,
            "image":      {"type": "base64", "value": _encode_frame(frame)},
            "confidence": CONF_THRESHOLD,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("predictions", [])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _T_base_cam(robot: SO101Follower, kin: SO101Kinematics) -> np.ndarray:
    obs = robot.get_observation()
    return kin.forward_kinematics(_joints_from_obs(obs)) @ T_EE_CAM


def _current_pos(robot: SO101Follower, kin: SO101Kinematics) -> np.ndarray:
    obs = robot.get_observation()
    return kin.forward_kinematics(_joints_from_obs(obs))[:3, 3].copy()


def _move(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target: np.ndarray,
    pid: bool = False,
    cancel_event: threading.Event | None = None,
) -> str:
    smooth_move(robot, kin, target)
    if cancel_event is not None and cancel_event.is_set():
        return "cancelled"
    return "done"


def _press_down(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    cancel_event: threading.Event | None = None,
) -> str:
    """
    Gentle key-press descent with stall detection.

    Phase 1: QP smooth-move approach to the press target.
    Phase 2: stall-detecting settle using a damped Jacobian pseudoinverse step;
             exits as soon as EE stalls (physical key contact).

    Returns 'contact', 'done', 'cancelled', or 'max_iter'.
    """
    stop = cancel_event if cancel_event is not None else threading.Event()

    # Phase 1 — approach
    smooth_move(robot, kin, target_pos)
    if stop.is_set():
        return "cancelled"

    # Phase 2 — stall-detecting settle with damped Jacobian pseudoinverse
    dt = 1.0 / FPS
    p_history: list[np.ndarray] = []

    for _ in range(_PRESS_MAX_ITER):
        if stop.is_set():
            return "cancelled"

        t0  = time.perf_counter()
        obs = robot.get_observation()
        q   = _joints_from_obs(obs)
        p   = kin.forward_kinematics(q)[:3, 3]

        error = target_pos - p
        dist  = float(np.linalg.norm(error))

        if dist < SETTLE_THRESHOLD_M:
            return "done"

        # Stall detection — stopped making progress → key contact
        p_history.append(p.copy())
        if len(p_history) > _STALL_WINDOW:
            p_history.pop(0)
        if len(p_history) == _STALL_WINDOW:
            travel = float(np.linalg.norm(p_history[-1] - p_history[0]))
            if travel < _STALL_MIN_M:
                print(f"  [press] Key contact  "
                      f"travel={travel*1000:.2f}mm / {_STALL_WINDOW} steps  "
                      f"residual={dist*1000:.1f}mm")
                return "contact"

        # Damped Jacobian pseudoinverse step (no IK library required)
        correction = np.clip(error, -0.006, 0.006)
        J = kin.position_jacobian(q, kin.active_joints)
        lam_sq = 0.0025
        J_damp = J.T @ np.linalg.inv(J @ J.T + lam_sq * np.eye(3))
        delta_q_deg = np.degrees(J_damp @ correction)
        for i, name in enumerate(kin.active_joints):
            q[JOINT_INDEX[name]] += delta_q_deg[i]
        q = kin.clip_joints(q)

        action = {f"{n}.pos": float(q[j]) for j, n in enumerate(MOTOR_NAMES)
                  if n != "gripper"}
        action["gripper.pos"] = obs["gripper.pos"]
        action["wrist_roll.pos"] = FIXED_WRIST_ROLL_DEG
        robot.send_action(action)

        time.sleep(max(dt - (time.perf_counter() - t0), 0.0))

    return "max_iter"


def _pnp_key_position(
    frame: np.ndarray,
    target_key: str,
    robot: SO101Follower,
    kin: SO101Kinematics,
    step_label: str,
) -> tuple[np.ndarray, float]:
    """
    Run detection + PnP on frame, return (key_pos_base, keyboard_z).

    keyboard_z is the PnP-derived Z of the key surface in robot base frame.
    Raises ValueError if detection or PnP fails.
    """
    t0    = time.perf_counter()
    preds = detect_keys(frame)
    print(f"  [{step_label}] Detection: {len(preds)} objects in "
          f"{(time.perf_counter()-t0)*1000:.0f} ms")

    dets = detections_from_roboflow(preds)
    if not dets:
        raise ValueError(f"[{step_label}] No keys detected.")

    T_bc = _T_base_cam(robot, kin)
    positions, reproj_err = get_key_positions_in_base_frame(
        dets, T_bc, CAMERA_K, DIST_COEFFS
    )
    print(f"  [{step_label}] PnP reproj error: {reproj_err:.2f} px")

    from keyboard_pnp import _normalize
    canonical = _normalize(target_key)
    if canonical not in positions:
        raise ValueError(
            f"[{step_label}] '{target_key}' not in German ISO layout."
        )

    key_pos   = positions[canonical]
    # Use the median Z of all keys as a robust surface estimate
    keyboard_z = float(np.median([p[2] for p in positions.values()]))
    print(f"  [{step_label}] '{target_key}' → "
          f"({key_pos[0]:+.4f}, {key_pos[1]:+.4f}, {key_pos[2]:+.4f}) m  "
          f"[surface z={keyboard_z:+.4f} m]")
    return key_pos, keyboard_z


# ─── Two-step press ───────────────────────────────────────────────────────────

def press_key(
    target_key: str,
    robot: SO101Follower,
    kin: SO101Kinematics,
    get_frame,
    lift: bool = True,
    cancel_event: threading.Event | None = None,
) -> None:
    """
    PnP-based coarse → fine key press.

    Parameters
    ----------
    target_key   : key label (e.g. 'a', 'enter', 'space')
    robot        : connected SO101Follower
    kin          : SO101Kinematics instance
    get_frame    : callable() → np.ndarray  (camera frame provider)
    lift         : if True, lift back to hover height after press
    cancel_event : optional threading.Event to abort the move

    Raises
    ------
    ValueError   if the key cannot be located via PnP in either step
    RuntimeError if camera read fails
    """

    # ── Step 1: Coarse — PnP from wide view, move to hover ───────────────────
    print(f"\n[Step 1/2 — coarse]  Locating '{target_key}' via PnP …")
    frame = get_frame()
    if frame is None:
        raise RuntimeError("Camera read failed.")

    key_pos_coarse, keyboard_z_coarse = _pnp_key_position(
        frame, target_key, robot, kin, "coarse"
    )

    hover_z  = keyboard_z_coarse + HOVER_OFFSET_M
    coarse_target = np.array([key_pos_coarse[0], key_pos_coarse[1], hover_z])

    # Clamp to workspace
    coarse_target = np.clip(coarse_target, WS_MIN, WS_MAX)
    print(f"  Hover target = ({coarse_target[0]:+.4f}, {coarse_target[1]:+.4f}, "
          f"{coarse_target[2]:+.4f}) m")

    result = _move(robot, kin, coarse_target, pid=False, cancel_event=cancel_event)
    print(f"  Coarse move: {result}")
    if result == "cancelled":
        return

    # ── Step 2: Fine — PnP from close view, press ────────────────────────────
    print(f"\n[Step 2/2 — fine]    Re-detecting '{target_key}' from close range …")
    time.sleep(0.3)   # let arm vibration damp out

    frame = get_frame()
    if frame is None:
        raise RuntimeError("Camera read failed (fine step).")

    key_pos_fine, keyboard_z_fine = _pnp_key_position(
        frame, target_key, robot, kin, "fine"
    )

    press_z = keyboard_z_fine - PRESS_BELOW_M
    fine_target = np.array([key_pos_fine[0], key_pos_fine[1], press_z])
    fine_target = np.clip(fine_target, WS_MIN, WS_MAX)
    print(f"  Press target = ({fine_target[0]:+.4f}, {fine_target[1]:+.4f}, "
          f"{fine_target[2]:+.4f}) m  (stall-detect descent)")

    # Use stall-detecting gentle descent — stops on key contact, no PID wind-up
    result = _press_down(robot, kin, fine_target, cancel_event)
    print(f"  Press result: {result}")
    if result == "cancelled":
        return

    # ── Lift ─────────────────────────────────────────────────────────────────
    if lift:
        lift_target    = fine_target.copy()
        lift_target[2] = keyboard_z_fine + HOVER_OFFSET_M
        lift_target    = np.clip(lift_target, WS_MIN, WS_MAX)
        _move(robot, kin, lift_target, pid=False, cancel_event=cancel_event)
        print(f"  Lifted to z={lift_target[2]:+.4f} m")

    print(f"\nDone — pressed '{target_key}'.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PnP-based two-step coarse→fine keyboard key press for SO-101."
    )
    parser.add_argument(
        "--key", required=True,
        help="Key label to press, e.g. 'a', 'enter', 'space'.",
    )
    parser.add_argument(
        "--no-lift", action="store_true",
        help="Skip lifting after press (leaves EE on the key).",
    )
    args = parser.parse_args()

    if not ROBOFLOW_API_KEY:
        raise RuntimeError(
            f"ROBOFLOW_API_KEY is not set.\nAdd it to  {_ENV_FILE}"
        )

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
        print(f"[warn] Could not open camera index {CAMERA_INDEX}")

    def _get_frame():
        ret, frame = cap.read()
        return frame if ret else None

    try:
        press_key(args.key, robot, kin, _get_frame, lift=not args.no_lift)
    except (ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}")
    finally:
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
