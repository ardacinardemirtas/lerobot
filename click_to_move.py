#!/usr/bin/env python
"""
click_to_move.py  —  Click on the live camera feed to move the SO-101 end effector.

Transformation pipeline:
    pixel (u, v)
        → undistort with DIST_COEFFS
        → camera ray  (via CAMERA_K intrinsics, scaled to 640×480)
        → robot base frame  (via FK(q) @ T_EE_CAM, eye-in-hand)
        → intersect horizontal plane z = TARGET_Z_M
        → smooth_move(x, y, TARGET_Z_M)  [smoothstep IK + PID settle]

Controls (OpenCV window):
    Left-click    move EE to point on work surface under cursor
    Right-click   cancel an in-progress move
    p             toggle PID correction in the settle loop
    s             print current EE position to terminal
    h             toggle help overlay
    q / Esc       return to start position and quit
"""

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ikpy.chain import Chain

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PORT            = "COM5"
ROBOT_ID        = "my_awesome_follower_arm"
CALIBRATION_DIR = Path(r"C:\Users\plata\robots")
URDF_PATH       = r"C:\Users\plata\robots\lerobot\calibration\so101_new_calib.urdf"
FPS             = 30

ARM_JOINTS  = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]
WS_MIN      = np.array([-0.35, -0.35, -0.10])
WS_MAX      = np.array([ 0.35,  0.35,  0.50])
MAX_MOVE_M  = 0.30

ROLL_FIXED_DEG = -90.0   # wrist roll held constant (same as move_to_position_old.py)

# Camera — lerobot-find-cameras reported index 1
CAMERA_INDEX  = 1
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

#calibration at 640x480
CAMERA_K = np.array([
    [341.4095, 0.0000, 329.1160],
    [0.0000, 340.6890, 219.6222],
    [0.0000, 0.0000, 1.0000],
], dtype=float)
# Distortion coefficients are dimensionless — stay the same across resolutions
DIST_COEFFS = np.array(
    [0.0799172, -0.15934891, -0.00045079, -0.00048855, 0.11357439],
    dtype=float,
)

# Hand-eye: 4×4 transform from camera frame → end-effector frame (P_ee = T_EE_CAM @ P_cam)
# Ground truth — verified independently by two solvers (PARK/HORAUD/DANIILIDIS + external).
T_EE_CAM = np.array([
    [-0.99938712, +0.02241607, +0.02688672, -0.00942822],
    [-0.03310399, -0.85490033, -0.51773502, +0.06081714],
    [+0.01137988, -0.51830777, +0.85511845, -0.03862474],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=float)

# Height of the work surface in robot base frame — table is at z = 0
TARGET_Z_M = 0

# Absolute home configuration — captured from the robot on 2026-05-09.
# Update by running:  python _read_home.py
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
HOME_DEG = np.array([-1.2308, -99.5604, 97.8901, 34.4615, -89.6264, 99.8606])

# ── smooth_move / settle parameters (from move_to_position_old.py) ────────────
KP = 1.0       # proportional
KI = 5.0       # integral — eliminates steady-state error
KD = 0.5       # derivative
SETTLE_THRESHOLD_M = 0.003
SETTLE_MAX_ITER    = 60
D_ALPHA            = 0.3   # derivative low-pass filter coefficient
SMOOTH_DURATION_S  = 2.0   # interpolation duration for smooth_move

# ─────────────────────────────────────────────────────────────────────────────


# ─── Kinematics ───────────────────────────────────────────────────────────────

class SO101Kinematics:
    """
    ikpy FK + IK for SO-101.  Matches move_to_position_old.py exactly:
    wrist_roll is passive and locked at ROLL_FIXED_DEG; 4 joints are active.
    """

    def __init__(self, urdf_path: str):
        _probe = Chain.from_urdf_file(urdf_path)
        self._link_names = [lnk.name for lnk in _probe.links]

        # 4 active joints; wrist_roll is passive (held at ROLL_FIXED_DEG)
        _active = [j for j in ARM_JOINTS if j != "wrist_roll"]
        mask = [name in _active for name in self._link_names]

        self.chain = Chain.from_urdf_file(urdf_path, active_links_mask=mask)
        self._arm_idx          = [self._link_names.index(n) for n in ARM_JOINTS]
        self._wrist_roll_idx   = self._link_names.index("wrist_roll")

        print("Kinematic chain:")
        for i, name in enumerate(self._link_names):
            print(f"  [{i:2d}] {name}" + (" ← active" if mask[i] else ""))
        print()

    def _to_ikpy(self, arm_deg: np.ndarray) -> np.ndarray:
        q = np.zeros(len(self._link_names))
        for idx, deg in zip(self._arm_idx, arm_deg):
            q[idx] = np.deg2rad(deg)
        return q

    def _from_ikpy(self, q: np.ndarray) -> np.ndarray:
        return np.array([np.rad2deg(q[i]) for i in self._arm_idx])

    def _clip_to_bounds(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        for i, link in enumerate(self.chain.links):
            if link.bounds is not None:
                lo, hi = link.bounds
                if lo is not None:
                    q[i] = max(q[i], lo)
                if hi is not None:
                    q[i] = min(q[i], hi)
        return q

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """6-element degree array → 4×4 world-to-EE transform."""
        return self.chain.forward_kinematics(self._to_ikpy(joint_pos_deg[:5]))

    def inverse_kinematics(
        self, current_deg: np.ndarray, target_T: np.ndarray
    ) -> np.ndarray:
        """
        Solve IK for target_T (4×4).  Wrist roll is held at ROLL_FIXED_DEG.
        Returns new 6-element degree array with gripper preserved.
        """
        initial = self._clip_to_bounds(self._to_ikpy(current_deg[:5]))
        initial[self._wrist_roll_idx] = np.deg2rad(ROLL_FIXED_DEG)

        result = self.chain.inverse_kinematics(
            target_position=target_T[:3, 3],
            initial_position=initial,
        )
        out = np.array(current_deg, dtype=float)
        out[:5] = self._from_ikpy(result)
        out[4]  = ROLL_FIXED_DEG   # enforce explicitly
        return out


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _joints_from_obs(obs: dict) -> np.ndarray:
    return np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=float)


def _clamp_target(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    target = np.clip(target, WS_MIN, WS_MAX)
    dist = float(np.linalg.norm(target - current))
    if dist > MAX_MOVE_M:
        target = current + (target - current) * (MAX_MOVE_M / dist)
    return target


# ─── Motion (from move_to_position_old.py) ────────────────────────────────────

def settle(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    target_T: np.ndarray,
    stop_event: threading.Event,
    pid_enabled: bool,
) -> str:
    """
    PID settle loop: drives FK-reported position error to zero.
    Returns 'done', 'cancelled', or 'max_iter'.
    """
    integral   = np.zeros(3)
    prev_error = np.zeros(3)
    deriv_filt = np.zeros(3)
    dt         = 1.0 / FPS

    for _ in range(SETTLE_MAX_ITER):
        if stop_event.is_set():
            return "cancelled"

        t0 = time.perf_counter()

        obs    = robot.get_observation()
        q      = _joints_from_obs(obs)
        T_curr = kin.forward_kinematics(q)
        p_curr = T_curr[:3, 3]

        error = target_pos - p_curr
        dist  = float(np.linalg.norm(error))

        if dist < SETTLE_THRESHOLD_M:
            return "done"

        if pid_enabled:
            integral  += error * dt
            raw_d      = (error - prev_error) / dt
            deriv_filt = D_ALPHA * raw_d + (1 - D_ALPHA) * deriv_filt
            correction = KP * error + KI * integral + KD * deriv_filt
            prev_error = error
        else:
            correction = error   # simple proportional only (KP implicitly 1)

        corr_norm = float(np.linalg.norm(correction))
        if corr_norm > 0.05:
            correction *= 0.05 / corr_norm

        T_cmd = target_T.copy()
        T_cmd[:3, 3] = p_curr + correction
        q = kin.inverse_kinematics(q, T_cmd)

        action = {f"{n}.pos": float(q[j]) for j, n in enumerate(MOTOR_NAMES) if n != "gripper"}
        action["gripper.pos"] = obs["gripper.pos"]
        robot.send_action(action)

        time.sleep(max(dt - (time.perf_counter() - t0), 0.0))

    return "max_iter"


def smooth_move(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    stop_event: threading.Event,
    pid_enabled: bool,
    duration_s: float = SMOOTH_DURATION_S,
) -> str:
    """
    Smoothstep Cartesian interpolation to target_pos, then PID settle.
    Orientation is held constant throughout.
    Returns 'done', 'cancelled', or 'max_iter'.
    """
    obs    = robot.get_observation()
    q      = _joints_from_obs(obs)
    T_start = kin.forward_kinematics(q)
    p_start = T_start[:3, 3].copy()
    gripper = obs["gripper.pos"]

    target_pos = _clamp_target(target_pos, p_start)
    dist = float(np.linalg.norm(target_pos - p_start))
    if dist < 1e-4:
        return "done"

    T_target = T_start.copy()
    T_target[:3, 3] = target_pos

    n_steps = max(int(duration_s * FPS), 1)

    for i in range(1, n_steps + 1):
        if stop_event.is_set():
            return "cancelled"

        t0 = time.perf_counter()

        alpha = i / n_steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)   # smoothstep easing

        T_wp = T_start.copy()
        T_wp[:3, 3] = (1.0 - alpha) * p_start + alpha * target_pos
        q = kin.inverse_kinematics(q, T_wp)

        action = {f"{n}.pos": float(q[j]) for j, n in enumerate(MOTOR_NAMES) if n != "gripper"}
        action["gripper.pos"] = gripper
        robot.send_action(action)

        time.sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    return settle(robot, kin, target_pos, T_target, stop_event, pid_enabled)


# ─── Coordinate mapping ───────────────────────────────────────────────────────

# Set to True to print full intermediate values on every click — for calibration debugging.
DEBUG_PROJECTION = True


def pixel_to_robot_target(
    u: float,
    v: float,
    T_base_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    z_target: float,
) -> Optional[np.ndarray]:
    """
    Back-project pixel (u, v) into a 3D point in robot base frame by intersecting
    the camera ray with the horizontal plane  z = z_target.

    Undistorts the pixel first so lens distortion does not skew the ray.
    Returns the 3D point or None if no valid intersection.
    """
    pt   = np.array([[[u, v]]], dtype=np.float32)
    pt_u = cv2.undistortPoints(pt, K, dist, P=K)   # P=K → result stays in pixel coords
    u_u  = float(pt_u[0, 0, 0])
    v_u  = float(pt_u[0, 0, 1])

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    d_cam  = np.array([(u_u - cx) / fx, (v_u - cy) / fy, 1.0])
    R      = T_base_cam[:3, :3]
    origin = T_base_cam[:3, 3]
    d_base = R @ d_cam

    if DEBUG_PROJECTION:
        import math
        d_cam_n  = d_cam / np.linalg.norm(d_cam)
        obliquity = math.acos(abs(d_cam_n[2])) * 180.0 / math.pi
        print("\n─── pixel_to_robot_target diagnostics ───")
        print(f"  pixel         : ({u}, {v})  undistorted: ({u_u:.2f}, {v_u:.2f})")
        print(f"  d_cam (norm)  : ({d_cam_n[0]:+.4f}, {d_cam_n[1]:+.4f}, {d_cam_n[2]:+.4f})"
              f"  obliquity={obliquity:.1f}°")
        print(f"  cam origin    : ({origin[0]:+.4f}, {origin[1]:+.4f}, {origin[2]:+.4f})")
        print(f"  d_base        : ({d_base[0]:+.4f}, {d_base[1]:+.4f}, {d_base[2]:+.4f})")

    if abs(d_base[2]) < 1e-6:
        if DEBUG_PROJECTION:
            print("  → ray parallel to work-surface plane (d_base[2] ≈ 0)")
        return None

    t = (z_target - origin[2]) / d_base[2]

    if DEBUG_PROJECTION:
        print(f"  ray param t   : {t:.4f}  (camera ht above plane = {origin[2]:.4f} m)")

    if t < 0:
        if DEBUG_PROJECTION:
            print("  → intersection behind camera (t < 0) — check T_EE_CAM or TARGET_Z_M")
        return None

    result = origin + t * d_base
    if DEBUG_PROJECTION:
        print(f"  result (base) : ({result[0]:+.4f}, {result[1]:+.4f}, {result[2]:+.4f})")
    return result


def reproject_ground_grid(
    T_base_cam: np.ndarray,
    K: np.ndarray,
    z_target: float,
    frame: np.ndarray,
    grid_step_m: float = 0.05,
) -> np.ndarray:
    """
    Overlay a grid of known ground-plane points (robot base frame, z=z_target)
    as reprojected dots on the frame.  Use this to visually verify T_base_cam.
    Press 'g' in the GUI to toggle it.
    """
    h, w = frame.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    T_cam_base = np.linalg.inv(T_base_cam)

    for x_m in np.arange(-0.30, 0.31, grid_step_m):
        for y_m in np.arange(-0.30, 0.31, grid_step_m):
            P_base = np.array([x_m, y_m, z_target, 1.0])
            P_cam  = T_cam_base @ P_base
            if P_cam[2] <= 0:
                continue
            u = fx * P_cam[0] / P_cam[2] + cx
            v = fy * P_cam[1] / P_cam[2] + cy
            if 0 <= u < w and 0 <= v < h:
                pu, pv = int(round(u)), int(round(v))
                cv2.circle(frame, (pu, pv), 3, (0, 255, 255), -1)
                label = f"{x_m:.0f},{y_m:.0f}"
                cv2.putText(frame, label, (pu + 4, pv - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 220, 220), 1)
    return frame


# ─── GUI ──────────────────────────────────────────────────────────────────────

_HELP = [
    "Left-click        move EE to cursor position on work surface",
    "Right-click       cancel current move",
    "Arrow Up/Down     jog +/- X by 0.02 m",
    "Arrow Left/Right  jog -/+ Y by 0.02 m",
    "Space             jog +Z by 0.02 m (up)",
    "r / HOME key      return to home position",
    "p                 toggle PID in the settle loop",
    "s                 print EE position (touch table first → verify z ≈ TARGET_Z_M)",
    "g                 toggle reprojection grid overlay (calibration debug)",
    "d                 toggle per-click projection diagnostics in terminal",
    "h                 toggle this help",
    "q / Esc           return home and quit",
]
_WIN    = "SO-101  Click-to-Move"
JOG_M   = 0.02   # metres per arrow-key press

# Windows virtual-key codes returned by cv2.waitKeyEx
_KEY_UP    = 2490368
_KEY_DOWN  = 2621440
_KEY_LEFT  = 2424832
_KEY_RIGHT = 2555904
_KEY_HOME  = 2359296
_KEY_SPACE = 32


class ClickToMoveApp:
    def __init__(
        self,
        robot: SO101Follower,
        kin: SO101Kinematics,
        cap: cv2.VideoCapture,
        home_pos: np.ndarray,
    ):
        self.robot    = robot
        self.kin      = kin
        self.cap      = cap
        self._home    = home_pos.copy()

        self._pid       = False
        self._help      = False
        self._grid      = False   # reprojection overlay (debug)
        self._status    = "Ready — left-click to move  |  arrows to jog  |  r = home"
        self._click_uv:  Optional[tuple[int, int]] = None
        self._target_3d: Optional[np.ndarray]      = None
        self._T_base_cam_last: Optional[np.ndarray] = None   # cached for grid overlay

        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock   = threading.Lock()

    # ── camera transform ──────────────────────────────────────────────────────

    def _T_base_cam(self) -> np.ndarray:
        """T_base_cam = FK(q) @ T_EE_CAM  (eye-in-hand)."""
        obs = self.robot.get_observation()
        q   = _joints_from_obs(obs)
        T_base_ee = self.kin.forward_kinematics(q)
        T = T_base_ee @ T_EE_CAM
        self._T_base_cam_last = T
        return T

    # ── mouse callback ────────────────────────────────────────────────────────

    # HOME button rect (set in _draw, checked in _on_mouse)
    _btn_home: Optional[tuple[int, int, int, int]] = None   # x1,y1,x2,y2

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check HOME button first
            if self._btn_home is not None:
                x1, y1, x2, y2 = self._btn_home
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._go_home()
                    return
            self._handle_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._stop.set()

    def _handle_click(self, u: int, v: int) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._status = "Already moving — right-click to cancel first"
                return

        try:
            T = self._T_base_cam()
        except Exception as exc:
            self._status = f"FK error: {exc}"
            return

        target = pixel_to_robot_target(u, v, T, CAMERA_K, DIST_COEFFS, TARGET_Z_M)
        if target is None:
            self._status = "Ray misses work surface — check T_EE_CAM / TARGET_Z_M"
            return

        if not (np.all(target >= WS_MIN) and np.all(target <= WS_MAX)):
            self._status = "Target outside workspace — will be clamped to bounds"

        with self._lock:
            self._click_uv  = (u, v)
            self._target_3d = target.copy()

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._move_worker, args=(target, "Click"), daemon=True
        )
        self._thread.start()

    # ── go home / jog ─────────────────────────────────────────────────────────

    def _go_home(self) -> None:
        """Cancel any running move and drive the EE back to the recorded home position."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._move_worker, args=(self._home, "Home"), daemon=True
        )
        self._thread.start()

    def _jog(self, dx: float = 0, dy: float = 0, dz: float = 0) -> None:
        """Relative move in robot-base frame. Cancels any current move first."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

        obs = self.robot.get_observation()
        q   = _joints_from_obs(obs)
        T   = self.kin.forward_kinematics(q)
        target = T[:3, 3].copy()
        target[0] += dx
        target[1] += dy
        target[2] += dz

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._move_worker, args=(target, "Jog"), daemon=True
        )
        self._thread.start()

    # ── move worker ───────────────────────────────────────────────────────────

    def _move_worker(self, target: np.ndarray, label: str = "Target") -> None:
        tgt = f"({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:.3f}) m"
        self._status = f"{label} → {tgt} …"
        result = smooth_move(self.robot, self.kin, target, self._stop, self._pid)
        self._status = {
            "done":     f"Arrived at {tgt}",
            "cancelled": "Move cancelled",
            "max_iter": f"Max settle iters — residual remains near {tgt}",
        }.get(result, result)

    # ── frame overlay ─────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # ── HOME button (top-right) ───────────────────────────────────────────
        btn_w, btn_h_px = 80, 28
        x1 = w - btn_w - 6;  x2 = w - 6
        y1 = 6;               y2 = y1 + btn_h_px
        self._btn_home = (x1, y1, x2, y2)
        moving = self._thread is not None and self._thread.is_alive()
        btn_col = (50, 130, 220) if not moving else (30, 80, 140)
        cv2.rectangle(frame, (x1, y1), (x2, y2), btn_col, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 210, 255), 1)
        cv2.putText(frame, "HOME [r]", (x1 + 6, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Status bar (bottom) ───────────────────────────────────────────────
        bar_h = 56
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)

        cv2.putText(frame, self._status, (8, h - bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        debug_flag = "DBG " if DEBUG_PROJECTION else ""
        grid_flag  = "GRID " if self._grid else ""
        info = (f"PID: {'ON' if self._pid else 'OFF'}  |  Z={TARGET_Z_M:.2f}m  |  "
                f"{debug_flag}{grid_flag}arrows=jog  space=up  r=home  g=grid  d=dbg  h=help  q=quit")
        cv2.putText(frame, info, (8, h - bar_h + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 220, 160), 1, cv2.LINE_AA)

        # ── Reprojection grid overlay (calibration debug) ─────────────────────
        if self._grid and self._T_base_cam_last is not None:
            try:
                reproject_ground_grid(self._T_base_cam_last, CAMERA_K, TARGET_Z_M, frame)
            except Exception:
                pass

        # ── Click crosshair ───────────────────────────────────────────────────
        if self._click_uv is not None:
            u, v = self._click_uv
            colour = (0, 150, 255) if moving else (50, 220, 50)

            cv2.circle(frame, (u, v), 11, colour, 2, cv2.LINE_AA)
            cv2.line(frame, (u - 16, v), (u + 16, v), colour, 1, cv2.LINE_AA)
            cv2.line(frame, (u, v - 16), (u, v + 16), colour, 1, cv2.LINE_AA)

            if self._target_3d is not None:
                t = self._target_3d
                cv2.putText(
                    frame,
                    f"({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:.3f}) m",
                    (u + 14, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA,
                )

        # ── Help overlay ──────────────────────────────────────────────────────
        if self._help:
            for i, line in enumerate(_HELP):
                cv2.putText(frame, line, (10, 28 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 200), 1, cv2.LINE_AA)

        return frame

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WIN, self._on_mouse)

        print(f"{'─'*55}")
        print("  SO-101 Click-to-Move GUI")
        print(f"  Camera index {CAMERA_INDEX}  {CAMERA_WIDTH}×{CAMERA_HEIGHT}")
        print(f"  Work-surface Z = {TARGET_Z_M:.3f} m  (wrist roll = {ROLL_FIXED_DEG}°)")
        print("  Left-click on the image to move.  Press [h] for help.")
        print(f"{'─'*55}\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "No camera signal", (80, CAMERA_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 200), 2)

            self._draw(frame)
            cv2.imshow(_WIN, frame)

            # waitKeyEx returns full key code (needed for arrow keys on Windows)
            key = cv2.waitKeyEx(1)
            if key == -1:
                continue

            # ── special / extended keys ───────────────────────────────────────
            if key == _KEY_UP:
                self._jog(dx=+JOG_M)
            elif key == _KEY_DOWN:
                self._jog(dx=-JOG_M)
            elif key == _KEY_LEFT:
                self._jog(dy=-JOG_M)
            elif key == _KEY_RIGHT:
                self._jog(dy=+JOG_M)
            elif key == _KEY_SPACE:
                self._jog(dz=+0.1)
            elif key == _KEY_HOME:
                self._go_home()

            # ── regular ASCII keys ────────────────────────────────────────────
            elif key & 0xFF in (ord("q"), 27):   # q or Esc
                break
            elif key & 0xFF == ord("r"):
                self._go_home()
            elif key & 0xFF == ord("p"):
                self._pid = not self._pid
                self._status = f"PID settle {'ON' if self._pid else 'OFF'}"
            elif key & 0xFF == ord("h"):
                self._help = not self._help
            elif key & 0xFF == ord("s"):
                obs = self.robot.get_observation()
                q   = _joints_from_obs(obs)
                T   = self.kin.forward_kinematics(q)
                p   = T[:3, 3]
                print(f"EE  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}"
                      f"  (TARGET_Z_M={TARGET_Z_M:.4f} — should match when touching table)")
                # Also refresh cached T_base_cam for grid overlay
                try:
                    self._T_base_cam_last = T @ T_EE_CAM
                except Exception:
                    pass
            elif key & 0xFF == ord("g"):
                self._grid = not self._grid
                if self._grid and self._T_base_cam_last is None:
                    try:
                        self._T_base_cam_last = self._T_base_cam()
                    except Exception:
                        pass
                self._status = f"Grid overlay {'ON' if self._grid else 'OFF'} — yellow dots = ground plane at Z={TARGET_Z_M:.2f}m"
            elif key & 0xFF == ord("d"):
                global DEBUG_PROJECTION
                DEBUG_PROJECTION = not DEBUG_PROJECTION
                self._status = f"Click diagnostics {'ON' if DEBUG_PROJECTION else 'OFF'}"

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        cv2.destroyAllWindows()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    kin = SO101Kinematics(URDF_PATH)

    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        use_degrees=True,
    )
    robot = SO101Follower(config)
    robot.connect()

    # Home position derived from hardcoded HOME_DEG — same across all runs
    home = kin.forward_kinematics(HOME_DEG)[:3, 3].copy()
    print(f"Home: x={home[0]:+.4f}  y={home[1]:+.4f}  z={home[2]:+.4f}")

    print("Moving to home position …")
    stop_dummy = threading.Event()
    smooth_move(robot, kin, home, stop_dummy, pid_enabled=False)
    print("At home.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {CAMERA_INDEX} opened at {actual_w}×{actual_h}")
        if actual_w != CAMERA_WIDTH or actual_h != CAMERA_HEIGHT:
            print(f"[warn] Requested {CAMERA_WIDTH}×{CAMERA_HEIGHT} "
                  f"but got {actual_w}×{actual_h} — intrinsics may need adjustment")
    else:
        print(f"[warn] Could not open camera index {CAMERA_INDEX}")

    app = ClickToMoveApp(robot, kin, cap, home_pos=home)
    try:
        app.run()
    finally:
        # app.run() already cancelled any in-flight move; go home then clean up
        print("\nReturning to home position …")
        stop_dummy = threading.Event()
        smooth_move(robot, kin, home, stop_dummy, pid_enabled=False)
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
