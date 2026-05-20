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
KB_HOME_HEIGHT_M = 0.26

# XY offset applied to the keyboard-centre position when computing KB_HOME.
# Positive X = further from robot base (toward keyboard / forward).
# Positive Y = left from robot's front.
# Tune these so the arm is well-extended in the forward direction at KB_HOME,
# giving the IK enough freedom to press keys without changing wrist orientation.
KB_HOME_X_OFFSET_M = 0.05
KB_HOME_Y_OFFSET_M = 0.0

# Step 1: height above the key for the intermediate observation position.
# High enough to see the full keyboard top-down (good PnP), far from the key.
INTERMEDIATE_OFFSET_M = 0.1

# Systematic XY correction applied to every press target to compensate for
# a fixed physical bias (e.g. camera/EE mounting offset, PnP skew).
# Positive X = further from robot base, positive Y = left from robot's front.
# If the arm consistently presses to the LEFT, try increasing KEY_Y_CORRECTION_M
# in small steps (0.003–0.008 m). If it presses too far forward/back, adjust X.
KEY_X_CORRECTION_M = -0.008
KEY_Y_CORRECTION_M = -0.007

# Step 2: height above the key for the final hover before pressing.
HOVER_OFFSET_M = 0.04

# Confidence threshold for fine detection at the intermediate position.
# Lower than CONF_THRESHOLD because we want as many key correspondences as
# possible for a robust PnP from the top-down view.
FINE_CONF_THRESHOLD = 0.35

# How far below the PnP-derived key-top surface to aim for the press.
# Stall detection stops descent at key contact regardless of this value.
PRESS_BELOW_M = 0.0007

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

        # Descend in Z while actively correcting XY drift so the EE stays
        # directly over the key and cannot slide forward/sideways during press.
        xy_err  = target_pos[:2] - p[:2]
        xy_step = np.clip(xy_err * 25.0, -0.003, 0.003)
        correction = np.array([xy_step[0], xy_step[1], np.clip(z_err, -0.003, 0.003)])
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


# ─── Wrist / joint helpers ────────────────────────────────────────────────────

# Full joint config (degrees) at KB_HOME, captured by find_kb_home().
# Used to restore the exact pose (including look-down wrist) after each press.
_kb_home_wrist_flex_deg: float = 0.0
_kb_home_q_deg: np.ndarray | None = None


def _apply_wrist_flex(robot: SO101Follower, target_flex_deg: float,
                      n_steps: int = 25) -> None:
    """Smoothly move wrist_flex to target_flex_deg, holding all other joints."""
    obs        = robot.get_observation()
    start_flex = float(obs["wrist_flex.pos"])
    if abs(start_flex - target_flex_deg) < 2.0:
        return
    gripper = float(obs["gripper.pos"])
    dt = 1.0 / FPS
    for i in range(n_steps):
        t0    = time.perf_counter()
        alpha = (i + 1) / n_steps
        flex  = start_flex + alpha * (target_flex_deg - start_flex)
        action = {
            "shoulder_pan.pos":  float(obs["shoulder_pan.pos"]),
            "shoulder_lift.pos": float(obs["shoulder_lift.pos"]),
            "elbow_flex.pos":    float(obs["elbow_flex.pos"]),
            "wrist_flex.pos":    flex,
            "wrist_roll.pos":    FIXED_WRIST_ROLL_DEG,
            "gripper.pos":       gripper,
        }
        robot.send_action(action)
        time.sleep(max(dt - (time.perf_counter() - t0), 0.0))


def _tilt_to_look_down(robot: SO101Follower) -> None:
    """Re-apply the wrist_flex angle discovered by find_kb_home()."""
    _apply_wrist_flex(robot, _kb_home_wrist_flex_deg)


def _move_to_joints(robot: SO101Follower, q_target_deg: np.ndarray,
                    duration: float = 3.0) -> None:
    """Joint-space move with minimum-jerk interpolation."""
    obs     = robot.get_observation()
    q_start = _joints_from_obs(obs)
    q_tgt   = np.asarray(q_target_deg, dtype=float)
    gripper = float(obs["gripper.pos"])
    dt      = 1.0 / FPS
    n       = max(1, int(duration * FPS))
    for i in range(n):
        t0  = time.perf_counter()
        tau = (i + 1) / n
        s   = 10*tau**3 - 15*tau**4 + 6*tau**5   # minimum-jerk
        q   = q_start + s * (q_tgt - q_start)
        action = {
            "shoulder_pan.pos":  float(q[JOINT_INDEX["shoulder_pan"]]),
            "shoulder_lift.pos": float(q[JOINT_INDEX["shoulder_lift"]]),
            "elbow_flex.pos":    float(q[JOINT_INDEX["elbow_flex"]]),
            "wrist_flex.pos":    float(q[JOINT_INDEX["wrist_flex"]]),
            "wrist_roll.pos":    FIXED_WRIST_ROLL_DEG,
            "gripper.pos":       gripper,
        }
        robot.send_action(action)
        time.sleep(max(dt - (time.perf_counter() - t0), 0.0))


def _find_wrist_flex_for_direction(
    kin: SO101Kinematics,
    q: np.ndarray,
    target_dir: np.ndarray,
) -> tuple[float, float]:
    """
    Scan the wrist_flex joint range and return the angle whose camera optical
    axis (camera-frame Z in world) best aligns with target_dir.

    Returns (best_wrist_flex_deg, alignment) where alignment is the dot product
    of the best camera-Z with the normalised target_dir.
    """
    spec = kin.joints.get("wrist_flex")
    lo   = float(np.degrees(spec.lower)) if spec and spec.lower is not None else -90.0
    hi   = float(np.degrees(spec.upper)) if spec and spec.upper is not None else  90.0
    t    = np.asarray(target_dir, dtype=float)
    t   /= max(float(np.linalg.norm(t)), 1e-9)

    best_angle = float(q[JOINT_INDEX["wrist_flex"]])
    best_dot   = -2.0
    for angle in np.linspace(lo, hi, 37):   # ≈ 5° steps
        q_t = q.copy()
        q_t[JOINT_INDEX["wrist_flex"]] = angle
        cam_z = (kin.forward_kinematics(q_t) @ T_EE_CAM)[:3, 2]
        dot   = float(np.dot(cam_z, t))
        if dot > best_dot:
            best_dot   = dot
            best_angle = angle
    return best_angle, best_dot


# ─── Keyboard home ───────────────────────────────────────────────────────────

def find_kb_home(
    robot: SO101Follower,
    kin: SO101Kinematics,
    get_frame,
    n_refine: int = 3,
) -> np.ndarray:
    """
    Find and lock in the KB_HOME position for this run.

    1. Moves arm to HOME_DEG (rough start — tune until keyboard is roughly
       visible in the camera from that joint configuration).
    2. Detects the keyboard via PnP and computes the ideal above-keyboard XYZ.
    3. Moves to that XYZ, then scans all wrist_flex angles to find the one
       whose camera optical axis points most directly at the keyboard centre.
    4. Applies the best wrist_flex and repeats (n_refine times) to converge.

    The resulting wrist_flex is stored in _kb_home_wrist_flex_deg and
    reapplied automatically by _tilt_to_look_down() after every key press.

    Raises RuntimeError if the keyboard is not visible from HOME_DEG.
    """
    global _kb_home_wrist_flex_deg

    # ── Step 1: go to rough start ─────────────────────────────────────────────
    print("  [KB_HOME] Moving to rough start (HOME_DEG) …")
    _move_to_joints(robot, HOME_DEG, duration=3.0)
    time.sleep(0.3)

    pos: np.ndarray | None = None

    for iteration in range(n_refine):
        print(f"  [KB_HOME] Refinement {iteration + 1}/{n_refine} …")

        # ── Detect keyboard ───────────────────────────────────────────────────
        frame = get_frame()
        if frame is None:
            raise RuntimeError("KB_HOME: camera read failed.")
        T_bc  = _T_base_cam(robot, kin)
        preds = detect_keys(frame, FINE_CONF_THRESHOLD)
        dets  = detections_from_roboflow(preds)

        if not dets:
            raise RuntimeError(
                "KB_HOME: no keys detected from HOME_DEG. "
                "Adjust HOME_DEG in click_to_move.py until the keyboard is "
                "visible in the camera from that position."
            )

        try:
            get_key_positions_in_base_frame(dets, T_bc, CAMERA_K, DIST_COEFFS)
        except ValueError as exc:
            if iteration == 0:
                raise RuntimeError(f"KB_HOME: PnP failed ({exc}).") from exc
            # Later iterations can rely on the cache from the previous pass.

        # ── Compute above-keyboard XYZ ────────────────────────────────────────
        pos = get_keyboard_home_position(KB_HOME_HEIGHT_M)
        if pos is None:
            raise RuntimeError("KB_HOME: could not locate keyboard centre.")
        pos[0] += KB_HOME_X_OFFSET_M
        pos[1] += KB_HOME_Y_OFFSET_M
        pos = np.clip(pos, WS_MIN, WS_MAX)
        print(f"  [KB_HOME] pos = ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m")

        # ── Move to above-keyboard XYZ ────────────────────────────────────────
        smooth_move(robot, kin, pos)
        time.sleep(0.15)

        # ── Find wrist_flex that makes the camera look straight down (world -Z) ─
        obs = robot.get_observation()
        q   = _joints_from_obs(obs)

        best_flex, alignment = _find_wrist_flex_for_direction(
            kin, q, np.array([0.0, 0.0, -1.0])
        )
        print(f"  [KB_HOME] wrist_flex = {best_flex:.1f}°  "
              f"cam-down alignment = {alignment:.3f}")

        # Closed-loop joint-space wrist move (replaces open-loop _apply_wrist_flex).
        # Must be the LAST step — smooth_move above already corrected XYZ, and any
        # re-running of smooth_move here would undo the wrist adjustment.
        q_wrist = q.copy()
        q_wrist[JOINT_INDEX["wrist_flex"]] = best_flex
        _move_to_joints(robot, q_wrist, duration=1.5)
        _kb_home_wrist_flex_deg = best_flex
        time.sleep(0.1)

        if alignment > 0.98:
            print(f"  [KB_HOME] Converged at iteration {iteration + 1}.")
            break

    assert pos is not None
    global _kb_home_q_deg
    _kb_home_q_deg = _joints_from_obs(robot.get_observation()).copy()
    print(f"  [KB_HOME] Locked in: pos=({pos[0]:+.4f}, {pos[1]:+.4f}, "
          f"{pos[2]:+.4f}) m  wrist_flex={_kb_home_wrist_flex_deg:.1f}°")
    return pos


def return_to_kb_home(robot: SO101Follower, duration: float = 2.0) -> bool:
    """
    Return to the KB_HOME joint configuration saved by find_kb_home().
    Uses joint-space interpolation so position and camera-down orientation
    are both restored reliably without any orientation-vs-position conflict.
    Returns False (and does nothing) if find_kb_home() has not been called.
    """
    if _kb_home_q_deg is None:
        return False
    _move_to_joints(robot, _kb_home_q_deg, duration=duration)
    return True


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
        key_pos_obs[0] + KEY_X_CORRECTION_M,
        key_pos_obs[1] + KEY_Y_CORRECTION_M,
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
    time.sleep(0.10)   # let arm vibration damp out

    frame = get_frame()
    if frame is None:
        raise RuntimeError("Camera read failed (hover step).")

    key_pos_fine = _pnp_key_position(
        frame, target_key, robot, kin, "hover", conf=FINE_CONF_THRESHOLD
    )

    hover_target = np.array([
        key_pos_fine[0] + KEY_X_CORRECTION_M,
        key_pos_fine[1] + KEY_Y_CORRECTION_M,
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
    time.sleep(0.02)  # brief settle before descent

    press_target = np.array([
        key_pos_fine[0] + KEY_X_CORRECTION_M,
        key_pos_fine[1] + KEY_Y_CORRECTION_M,
        key_pos_fine[2] - PRESS_BELOW_M,
    ])
    press_target = np.clip(press_target, WS_MIN, WS_MAX)
    print(f"  Press target = ({press_target[0]:+.4f}, {press_target[1]:+.4f}, "
          f"{press_target[2]:+.4f}) m")

    result = _press_down(robot, kin, press_target, cancel_event)
    print(f"  Press result: {result}")
    if result == "cancelled":
        return

    # ── Return to KB_HOME (reverse path) ─────────────────────────────────────
    if lift:
        # Retrace the forward path in reverse: press → hover → obs → kb_home.
        # This lifts straight up above the key before any horizontal movement,
        # keeping the return trajectory safe and deterministic.
        print(f"  [lift 1/3] press → hover")
        result = _move(robot, kin, hover_target, cancel_event=cancel_event)
        if result == "cancelled":
            return

        print(f"  [lift 2/3] hover → obs")
        result = _move(robot, kin, obs_target, cancel_event=cancel_event)
        if result == "cancelled":
            return

        kb_home = get_keyboard_home_position(KB_HOME_HEIGHT_M)
        if kb_home is not None:
            lift_target = np.clip(kb_home, WS_MIN, WS_MAX)
            print(f"  [lift 3/3] obs → KB_HOME ({lift_target[0]:+.4f}, "
                  f"{lift_target[1]:+.4f}, {lift_target[2]:+.4f}) m")
            result = _move(robot, kin, lift_target, cancel_event=cancel_event)
            if result == "cancelled":
                return
            # Restore exact joint config from find_kb_home() — guarantees the
            # camera returns to the same look-down orientation every time.
            if _kb_home_q_deg is not None:
                _move_to_joints(robot, _kb_home_q_deg, duration=1.0)
            else:
                _tilt_to_look_down(robot)
        else:
            # No KB_HOME cache — just stay at obs height above the key.
            print(f"  [lift 3/3] no KB_HOME cache, holding at obs position")

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
