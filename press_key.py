#!/usr/bin/env python3
"""
press_key.py — Two-step keyboard key pressing for SO-101.

Pipeline
--------
Step 1  (Coarse)
    Capture frame from wrist camera in wide view.
    Run Roboflow key-detection model.
    Locate target key in pixel space.
    Apply distance-from-centre damping to compensate for off-axis inaccuracy.
    Move EE to (x_coarse, y_coarse, HOVER_Z) — hovering above the keyboard,
    so the camera is now zoomed in on the key region.

Step 2  (Fine)
    Capture a fresh frame from the closer vantage point.
    Detect target key again (more accurate from close range).
    Move EE to (x_fine, y_fine, PRESS_Z) to press the key.
    Lift back to HOVER_Z.

Usage
-----
    python press_key.py --key a
    python press_key.py --key enter --no-lift

Requires
--------
    - Roboflow inference server running (docker / local)
    - ROBOFLOW_API_KEY in env or keyboard_detection/.env.inference
    - Robot connected on PORT (see click_to_move.py CONFIG)
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

# ── Load .env.inference before resolving env vars ─────────────────────────────
_ENV_FILE = Path(__file__).parent / "keyboard_detection" / ".env.inference"
if _ENV_FILE.exists():
    for _ln in _ENV_FILE.read_text().splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Re-use robot infrastructure from click_to_move.py ─────────────────────────
from click_to_move import (  # noqa: E402
    SO101Kinematics,
    smooth_move,
    pixel_to_robot_target,
    _joints_from_obs,
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
    TARGET_Z_M as PRESS_Z,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Height at which the coarse move stops and the fine detection happens.
# Lower = more zoomed-in in step 2; must stay above PRESS_Z.
HOVER_Z = 0.06   # metres above table surface

# Roboflow inference
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST",    "http://localhost:9001")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY",  "")
MODEL_ID         = os.environ.get("ROBOFLOW_MODEL_ID", "keyboard-key-recognition-kw7nc/14")
CONF_THRESHOLD   = 0.50

# Damping for off-centre detections.
# At the image corner the observed overshoot was ~1.5×, so damp to 1/1.5 ≈ 0.67.
# Keys near the image centre get damp = 1.0 (no reduction).
DAMP_EDGE = 1.0 / 1.5   # ≈ 0.67 — applied at maximum pixel distance from centre

# ─── Detection ────────────────────────────────────────────────────────────────

def _encode_frame(frame: np.ndarray) -> str:
    """BGR OpenCV frame → base64 JPEG string."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def detect_keys(frame: np.ndarray) -> list[dict]:
    """
    POST frame to Roboflow inference server.
    Returns list of prediction dicts with keys: x, y, width, height, class, confidence.
    Pixel coordinates are in the input frame's pixel space.
    """
    resp = requests.post(
        f"{INFERENCE_HOST.rstrip('/')}/infer/object_detection",
        json={
            "id": str(uuid.uuid4()),
            "model_id": MODEL_ID,
            "api_key": ROBOFLOW_API_KEY,
            "image": {"type": "base64", "value": _encode_frame(frame)},
            "confidence": CONF_THRESHOLD,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("predictions", [])


def find_key(preds: list[dict], target: str) -> dict | None:
    """Return highest-confidence prediction matching target (case-insensitive)."""
    matches = [p for p in preds if p["class"].lower() == target.lower()]
    return max(matches, key=lambda p: p["confidence"]) if matches else None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _damp(u: float, v: float) -> float:
    """
    Linear distance-from-centre damping factor in [DAMP_EDGE, 1.0].

    Centre pixel  → 1.0  (full predicted displacement applied)
    Image corner  → DAMP_EDGE (~0.67, compensates 1.5× overshoot at the edge)
    """
    cx, cy   = CAMERA_WIDTH / 2.0, CAMERA_HEIGHT / 2.0
    max_dist = (cx ** 2 + cy ** 2) ** 0.5
    dist     = ((u - cx) ** 2 + (v - cy) ** 2) ** 0.5
    t        = min(dist / max_dist, 1.0)
    return 1.0 + (DAMP_EDGE - 1.0) * t


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
    stop = cancel_event if cancel_event is not None else threading.Event()
    return smooth_move(robot, kin, target, stop, pid_enabled=pid)


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
    Coarse → fine press of `target_key`.

    Raises
    ------
    ValueError   if the key cannot be detected in either step.
    RuntimeError if camera read or ray-plane intersection fails.
    """

    # ── Step 1: Coarse detection & hover move ─────────────────────────────────
    print(f"\n[Step 1/2 — coarse]  Detecting '{target_key}' in wide view …")
    frame = get_frame()
    if frame is None:
        raise RuntimeError("Camera read failed.")

    t0    = time.perf_counter()
    preds = detect_keys(frame)
    print(f"  Detection: {len(preds)} objects in {(time.perf_counter()-t0)*1000:.0f} ms")

    pred = find_key(preds, target_key)
    if pred is None:
        avail = sorted({p["class"] for p in preds})
        raise ValueError(f"'{target_key}' not found in wide view. Detected: {avail}")

    u, v = float(pred["x"]), float(pred["y"])
    damp = _damp(u, v)
    print(f"  Pixel=({u:.0f}, {v:.0f})  conf={pred['confidence']:.2f}  damp={damp:.2f}")

    T_bc  = _T_base_cam(robot, kin)
    pt    = pixel_to_robot_target(u, v, T_bc, CAMERA_K, DIST_COEFFS, PRESS_Z)
    if pt is None:
        raise RuntimeError("Ray-plane intersection failed in coarse step.")

    curr   = _current_pos(robot, kin)
    coarse = np.array([
        curr[0] + damp * (pt[0] - curr[0]),
        curr[1] + damp * (pt[1] - curr[1]),
        HOVER_Z,
    ])
    print(f"  Coarse target = ({coarse[0]:+.3f}, {coarse[1]:+.3f}, {coarse[2]:.3f}) m")
    result = _move(robot, kin, coarse, pid=False, cancel_event=cancel_event)
    print(f"  Motion result: {result}")

    # ── Step 2: Fine detection & press ────────────────────────────────────────
    print(f"\n[Step 2/2 — fine]    Detecting '{target_key}' in close view …")
    time.sleep(0.3)   # let arm vibration damp out before grabbing frame

    frame = get_frame()
    if frame is None:
        raise RuntimeError("Camera read failed (fine step).")

    t0    = time.perf_counter()
    preds = detect_keys(frame)
    print(f"  Detection: {len(preds)} objects in {(time.perf_counter()-t0)*1000:.0f} ms")

    pred = find_key(preds, target_key)
    if pred is None:
        avail = sorted({p["class"] for p in preds})
        raise ValueError(
            f"'{target_key}' not found in close view. "
            f"Detected: {avail}\n"
            f"Try lowering CONF_THRESHOLD or increasing HOVER_Z."
        )

    u, v = float(pred["x"]), float(pred["y"])
    print(f"  Pixel=({u:.0f}, {v:.0f})  conf={pred['confidence']:.2f}")

    T_bc   = _T_base_cam(robot, kin)
    target = pixel_to_robot_target(u, v, T_bc, CAMERA_K, DIST_COEFFS, PRESS_Z)
    if target is None:
        raise RuntimeError("Ray-plane intersection failed in fine step.")

    print(f"  Fine target  = ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:.3f}) m")
    result = _move(robot, kin, target, pid=True, cancel_event=cancel_event)
    print(f"  Motion result: {result}")

    # ── Lift back to hover height ─────────────────────────────────────────────
    if lift:
        lift_target    = target.copy()
        lift_target[2] = HOVER_Z
        _move(robot, kin, lift_target, pid=False, cancel_event=cancel_event)
        print(f"  Lifted to z={HOVER_Z:.3f} m")

    print(f"\nDone — pressed '{target_key}'.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-step coarse→fine keyboard key press for SO-101."
    )
    parser.add_argument(
        "--key", required=True,
        help="Key label to press, e.g. 'a', 'enter', 'space'.",
    )
    parser.add_argument(
        "--no-lift", action="store_true",
        help="Skip lifting the EE after pressing (leaves it on the key).",
    )
    args = parser.parse_args()

    if not ROBOFLOW_API_KEY:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set.\n"
            "Export it as an environment variable, or add it to\n"
            f"  {_ENV_FILE}"
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
