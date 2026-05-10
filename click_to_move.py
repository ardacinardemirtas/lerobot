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

# Original calibration was at 1920×1080.
# Camera crops 1920→1440 (center, 4:3) then scales 1440×1080→640×480.
# Uniform scale = 640/1440 = 4/9.  cx offset = (1920−1440)/2 = 240 px before scaling.
_s   = 640 / 1440          # = 4/9 ≈ 0.4444  (uniform for both axes)
_cx0 = 981.92182023 - 240  # subtract half-crop before scaling
CAMERA_K = np.array([
    [693.35550704 * _s,  0.0,                 _cx0              * _s],
    [0.0,                692.62105461 * _s,   496.33606080      * _s],
    [0.0,                0.0,                 1.0                   ],
], dtype=float)
# Distortion coefficients are dimensionless — stay the same across resolutions
DIST_COEFFS = np.array(
    [0.06046540, -0.07201475, -0.00076385, 0.00059471, -0.00305710],
    dtype=float,
)

# Hand-eye: 4×4 transform from camera frame → end-effector frame (P_ee = T_EE_CAM @ P_cam)
_R_cam_in_ee = np.array([
    [-0.99926494,  0.01263551,  0.03619287],
    [-0.02597731, -0.91749060, -0.39690828],
    [ 0.02819148, -0.39755672,  0.91714442],
], dtype=float)
_t_cam_in_ee = np.array([0.00348327, 0.08292417, -0.08243772], dtype=float)
T_EE_CAM = np.eye(4, dtype=float)
T_EE_CAM[:3, :3] = _R_cam_in_ee
T_EE_CAM[:3,  3] = _t_cam_in_ee

# Height of the work surface in robot base frame — table is at z = 0
TARGET_Z_M = 0

# Absolute home configuration — captured from the robot on 2026-05-09.
# Update by running:  python _read_home.py
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
HOME_DEG = np.array([-1.2308, -99.5604, 97.8901, 34.4615, -89.6264, 99.8606])

# ── smooth_move / settle parameters (from move_to_position_old.py) ────────────
KP = 1.0       # proportional
KI = 5.0       # integral — eliminates steady-state error
KD = 0.0       # derivative
SETTLE_THRESHOLD_M = 0.003
SETTLE_MAX_ITER    = 60
D_ALPHA            = 0.3   # derivative low-pass filter coefficient
SMOOTH_DURATION_S  = 2.0   # interpolation duration for smooth_move
SMOOTH_HZ          = 60    # command rate during trajectory streaming

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
    IK is pre-computed for all waypoints so the streaming loop has no solver
    latency — this keeps the command rate steady and prevents the servo from
    parking at each waypoint before the next command arrives.
    Returns 'done', 'cancelled', or 'max_iter'.
    """
    obs     = robot.get_observation()
    q       = _joints_from_obs(obs)
    T_start = kin.forward_kinematics(q)
    p_start = T_start[:3, 3].copy()
    gripper = obs["gripper.pos"]

    target_pos = _clamp_target(target_pos, p_start)
    dist = float(np.linalg.norm(target_pos - p_start))
    if dist < 1e-4:
        return "done"

    T_target = T_start.copy()
    T_target[:3, 3] = target_pos

    n_steps = max(int(duration_s * SMOOTH_HZ), 1)

    # Pre-compute all IK solutions so the streaming loop is latency-free.
    waypoint_joints: list[np.ndarray] = []
    q_wp = q.copy()
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)   # smoothstep easing
        T_wp = T_start.copy()
        T_wp[:3, 3] = (1.0 - alpha) * p_start + alpha * target_pos
        q_wp = kin.inverse_kinematics(q_wp, T_wp)
        waypoint_joints.append(q_wp.copy())

    dt = 1.0 / SMOOTH_HZ
    for q_cmd in waypoint_joints:
        if stop_event.is_set():
            return "cancelled"
        t0 = time.perf_counter()
        action = {f"{n}.pos": float(q_cmd[j]) for j, n in enumerate(MOTOR_NAMES) if n != "gripper"}
        action["gripper.pos"] = gripper
        robot.send_action(action)
        time.sleep(max(dt - (time.perf_counter() - t0), 0.0))

    return settle(robot, kin, target_pos, T_target, stop_event, pid_enabled)


# ─── Coordinate mapping ───────────────────────────────────────────────────────

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

    if abs(d_base[2]) < 1e-6:
        return None   # ray parallel to work-surface plane

    t = (z_target - origin[2]) / d_base[2]
    if t < 0:
        return None   # intersection behind camera

    return origin + t * d_base


# ─── GUI ──────────────────────────────────────────────────────────────────────

_HELP = [
    "Left-click        move EE to cursor position on work surface",
    "Right-click       cancel current move",
    "Arrow Up/Down     jog +/- X by 0.02 m",
    "Arrow Left/Right  jog -/+ Y by 0.02 m",
    "Space             jog +Z by 0.02 m (up)",
    "r / HOME key      return to home position",
    "p                 toggle PID in the settle loop",
    "s                 print EE position to terminal",
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
        self._status    = "Ready — left-click to move  |  arrows to jog  |  r = home"
        self._click_uv:  Optional[tuple[int, int]] = None
        self._target_3d: Optional[np.ndarray]      = None

        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock   = threading.Lock()

    # ── camera transform ──────────────────────────────────────────────────────

    def _T_base_cam(self) -> np.ndarray:
        """T_base_cam = FK(q) @ T_EE_CAM  (eye-in-hand)."""
        obs = self.robot.get_observation()
        q   = _joints_from_obs(obs)
        T_base_ee = self.kin.forward_kinematics(q)
        return T_base_ee @ T_EE_CAM

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

        info = (f"PID: {'ON' if self._pid else 'OFF'}  |  Z={TARGET_Z_M:.2f}m  |  "
                f"arrows=jog  space=up  r=home  h=help  q=quit")
        cv2.putText(frame, info, (8, h - bar_h + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 220, 160), 1, cv2.LINE_AA)

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
                print(f"EE  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}")

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
