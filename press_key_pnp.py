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
    get_keyboard_home_position,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Height above the keyboard centre for the dedicated "keyboard home" position.
# From here the whole keyboard fits in frame for maximum PnP correspondences.
KB_HOME_HEIGHT_M = 0.22

# Step 1: height above the key for the intermediate observation position.
# High enough to see the full keyboard top-down (good PnP), far from the key.
INTERMEDIATE_OFFSET_M = 0.10

# Step 2: height above the key for the final hover before pressing.
HOVER_OFFSET_M = 0.04

# Confidence threshold for fine detection at the intermediate position.
# Lower than CONF_THRESHOLD because we want as many key correspondences as
# possible for a robust PnP from the top-down view.
FINE_CONF_THRESHOLD = 0.35

# How far below the PnP-derived key-top surface to aim for the press.
# Stall detection stops descent at key contact regardless of this value.
PRESS_BELOW_M = 0.003

# Press-settle stall detection
# If the EE moves less than STALL_MIN_M over STALL_WINDOW consecutive steps
# it has made contact with the key and we stop immediately.
_STALL_WINDOW   = 5      # ~0.08 s at 60 Hz — detect contact quickly
_STALL_MIN_M    = 0.0003 # < 0.3 mm travel → contact
_PRESS_MAX_ITER = 80     # hard cap (~1.3 s)

# Roboflow inference
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST",    "http://localhost:9001")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY",  "")
MODEL_ID         = os.environ.get("ROBOFLOW_MODEL_ID", "keyboard-key-recognition-kw7nc/14")
CONF_THRESHOLD   = 0.50

# Minimum PnP inliers required to trust a pose estimate
MIN_PNP_INLIERS = 4

# If reproj error exceeds this AND no cached keyboard pose is available,
# the estimate is too uncertain — abort before moving the arm.
MAX_REPROJ_ACCEPT_PX = 5.0

# ─── Detection ────────────────────────────────────────────────────────────────

def _encode_frame(frame: np.ndarray) -> str:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def detect_keys(frame: np.ndarray, conf: float = CONF_THRESHOLD) -> list[dict]:
    """POST frame to Roboflow, return prediction list."""
    resp = requests.post(
        f"{INFERENCE_HOST.rstrip('/')}/infer/object_detection",
        json={
            "id":         str(uuid.uuid4()),
            "model_id":   MODEL_ID,
            "api_key":    ROBOFLOW_API_KEY,
            "image":      {"type": "base64", "value": _encode_frame(frame)},
            "confidence": conf,
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
    result = smooth_move(robot, kin, target)
    if cancel_event is not None and cancel_event.is_set():
        return "cancelled"
    if not result["success"] and result["final_error_mm"] > 20.0:
        raise RuntimeError(
            f"Move failed: arm stuck {result['final_error_mm']:.1f} mm from target "
            "(target may be outside reachable workspace)."
        )
    return "done"


def _press_down(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    cancel_event: threading.Event | None = None,
) -> str:
    """
    Gentle key-press descent with stall detection.

    Descends from the current hover position toward target_pos using a
    Z-only damped Jacobian step. XY is never commanded to change, which
    prevents the end-effector from drifting forward during the press.
    Stops as soon as stall (key contact) is detected or Z target is reached.

    Returns 'contact', 'done', 'cancelled', or 'max_iter'.
    """
    stop = cancel_event if cancel_event is not None else threading.Event()

    dt = 1.0 / FPS
    p_history: list[np.ndarray] = []

    for _ in range(_PRESS_MAX_ITER):
        if stop.is_set():
            return "cancelled"

        t0  = time.perf_counter()
        obs = robot.get_observation()
        q   = _joints_from_obs(obs)
        p   = kin.forward_kinematics(q)[:3, 3]

        z_err = float(target_pos[2] - p[2])

        if abs(z_err) < SETTLE_THRESHOLD_M:
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
                      f"z_residual={abs(z_err)*1000:.1f}mm")
                return "contact"

        # Z-only Jacobian step — drives arm straight down, no XY correction
        # so the end-effector cannot drift forward during the press.
        z_correction = np.array([0.0, 0.0, np.clip(z_err, -0.003, 0.003)])
        J = kin.position_jacobian(q, kin.active_joints)
        lam_sq = 0.0025
        J_damp = J.T @ np.linalg.inv(J @ J.T + lam_sq * np.eye(3))
        delta_q_deg = np.degrees(J_damp @ z_correction)
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
    conf: float = CONF_THRESHOLD,
) -> np.ndarray:
    """
    Run detection + PnP on frame, return key_pos_base (3,) in robot base frame.

    T_base_cam is read BEFORE the network call so the FK matches the captured
    frame even if the API call takes several seconds.
    Raises ValueError if detection or PnP fails.
    """
    # Read FK now, while the arm is settled and the frame was just captured.
    T_bc = _T_base_cam(robot, kin)

    t0    = time.perf_counter()
    preds = detect_keys(frame, conf)
    print(f"  [{step_label}] Detection: {len(preds)} objects in "
          f"{(time.perf_counter()-t0)*1000:.0f} ms")

    dets = detections_from_roboflow(preds)
    if not dets:
        raise ValueError(f"[{step_label}] No keys detected.")

    positions, reproj_err = get_key_positions_in_base_frame(
        dets, T_bc, CAMERA_K, DIST_COEFFS
    )
    print(f"  [{step_label}] PnP reproj error: {reproj_err:.2f} px")

    if reproj_err > MAX_REPROJ_ACCEPT_PX:
        raise ValueError(
            f"[{step_label}] PnP reprojection error {reproj_err:.2f} px "
            f"exceeds {MAX_REPROJ_ACCEPT_PX} px and no cached pose is available. "
            "Move to a position where more keys are visible and try again."
        )

    from keyboard_pnp import _normalize
    canonical = _normalize(target_key)
    if canonical not in positions:
        raise ValueError(
            f"[{step_label}] '{target_key}' not in German ISO layout."
        )

    key_pos = positions[canonical]
    print(f"  [{step_label}] '{target_key}' → "
          f"({key_pos[0]:+.4f}, {key_pos[1]:+.4f}, {key_pos[2]:+.4f}) m")
    return key_pos


# ─── Keyboard home ───────────────────────────────────────────────────────────

def find_kb_home(
    robot: SO101Follower,
    kin: SO101Kinematics,
    get_frame,
) -> np.ndarray:
    """
    Return the robot base frame position above the keyboard centre at
    KB_HOME_HEIGHT_M.  Uses the cached keyboard pose when available; otherwise
    runs a fresh detection from the current arm position to build the cache.

    Raises RuntimeError if the keyboard cannot be located.
    """
    pos = get_keyboard_home_position(KB_HOME_HEIGHT_M)
    if pos is not None:
        print(f"  [KB_HOME] cached → ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m")
        return np.clip(pos, WS_MIN, WS_MAX)

    # Cache is empty — run one detection to populate it.
    frame = get_frame()
    if frame is None:
        raise RuntimeError("KB_HOME: camera read failed.")
    T_bc  = _T_base_cam(robot, kin)
    preds = detect_keys(frame, FINE_CONF_THRESHOLD)
    dets  = detections_from_roboflow(preds)
    if dets:
        try:
            get_key_positions_in_base_frame(dets, T_bc, CAMERA_K, DIST_COEFFS)
        except ValueError:
            pass

    pos = get_keyboard_home_position(KB_HOME_HEIGHT_M)
    if pos is None:
        raise RuntimeError(
            "KB_HOME: keyboard not found. Move to a position where the "
            "keyboard is visible and try again."
        )
    print(f"  [KB_HOME] detected → ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m")
    return np.clip(pos, WS_MIN, WS_MAX)


# ─── Three-step press ─────────────────────────────────────────────────────────

def press_key(
    target_key: str,
    robot: SO101Follower,
    kin: SO101Kinematics,
    get_frame,
    lift: bool = True,
    cancel_event: threading.Event | None = None,
) -> None:
    """
    Three-step PnP key press: observe → hover → press.

    Step 1  (Observe)
        PnP from the current position → move to INTERMEDIATE_OFFSET_M above
        the key.  This gives a near-top-down camera view with the full keyboard
        visible — ideal geometry for PnP.

    Step 2  (Hover)
        Stabilise, re-detect with a lower confidence threshold (more keys
        matched from the top-down view), refine the key position →
        smooth_move to HOVER_OFFSET_M above the key.

    Step 3  (Press)
        Z-only stall-detecting Jacobian descent to the key surface.

    Parameters
    ----------
    target_key   : key label (e.g. 'a', 'enter', 'space')
    robot        : connected SO101Follower
    kin          : SO101Kinematics instance
    get_frame    : callable() → np.ndarray  (camera frame provider)
    lift         : if True, lift back to hover height after press
    cancel_event : optional threading.Event to abort the move
    """

    # ── Step 1: Observe — PnP from current position, move to intermediate ────
    print(f"\n[Step 1/3 — observe]  Locating '{target_key}' via PnP …")
    frame = get_frame()
    if frame is None:
        raise RuntimeError("Camera read failed.")

    key_pos_obs = _pnp_key_position(
        frame, target_key, robot, kin, "observe"
    )

    # Intermediate: directly above the key but high enough for a top-down view.
    obs_target = np.array([
        key_pos_obs[0],
        key_pos_obs[1],
        key_pos_obs[2] + INTERMEDIATE_OFFSET_M,
    ])
    obs_target = np.clip(obs_target, WS_MIN, WS_MAX)
    print(f"  Intermediate target = ({obs_target[0]:+.4f}, {obs_target[1]:+.4f}, "
          f"{obs_target[2]:+.4f}) m")

    result = _move(robot, kin, obs_target, cancel_event=cancel_event)
    print(f"  Observe move: {result}")
    if result == "cancelled":
        return

    # ── Step 2: Hover — re-detect from top-down view, move to hover height ───
    print(f"\n[Step 2/3 — hover]    Re-detecting '{target_key}' from top-down view …")
    time.sleep(0.2)   # let arm vibration damp out

    frame = get_frame()
    if frame is None:
        raise RuntimeError("Camera read failed (hover step).")

    key_pos_fine = _pnp_key_position(
        frame, target_key, robot, kin, "hover", conf=FINE_CONF_THRESHOLD
    )

    hover_target = np.array([
        key_pos_fine[0],
        key_pos_fine[1],
        key_pos_fine[2] + HOVER_OFFSET_M,
    ])
    hover_target = np.clip(hover_target, WS_MIN, WS_MAX)
    print(f"  Hover target = ({hover_target[0]:+.4f}, {hover_target[1]:+.4f}, "
          f"{hover_target[2]:+.4f}) m")

    result = _move(robot, kin, hover_target, cancel_event=cancel_event)
    print(f"  Hover move: {result}")
    if result == "cancelled":
        return

    # ── Step 3: Press — stall-detecting Z-only descent ───────────────────────
    print(f"\n[Step 3/3 — press]    Pressing '{target_key}' …")
    time.sleep(0.05)  # brief settle before descent

    press_target = np.array([
        key_pos_fine[0],
        key_pos_fine[1],
        key_pos_fine[2] - PRESS_BELOW_M,
    ])
    press_target = np.clip(press_target, WS_MIN, WS_MAX)
    print(f"  Press target = ({press_target[0]:+.4f}, {press_target[1]:+.4f}, "
          f"{press_target[2]:+.4f}) m")

    result = _press_down(robot, kin, press_target, cancel_event)
    print(f"  Press result: {result}")
    if result == "cancelled":
        return

    # ── Return to KB_HOME ────────────────────────────────────────────────────
    if lift:
        kb_home = get_keyboard_home_position(KB_HOME_HEIGHT_M)
        if kb_home is not None:
            lift_target = np.clip(kb_home, WS_MIN, WS_MAX)
            print(f"  Returning to KB_HOME ({lift_target[0]:+.4f}, "
                  f"{lift_target[1]:+.4f}, {lift_target[2]:+.4f}) m")
        else:
            lift_target    = press_target.copy()
            lift_target[2] = key_pos_fine[2] + HOVER_OFFSET_M
            lift_target    = np.clip(lift_target, WS_MIN, WS_MAX)
            print(f"  Lifting to hover z={lift_target[2]:+.4f} m (no KB_HOME cache)")
        _move(robot, kin, lift_target, cancel_event=cancel_event)

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
