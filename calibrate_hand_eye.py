#!/usr/bin/env python
"""
calibrate_hand_eye.py — Collect image + end-effector pose pairs for hand-eye calibration.

Workflow:
  1. Jog the robot to a new pose with the keyboard controls below.
  2. Hold a calibration target (checkerboard / ArUco board) in front of the camera.
  3. Press SPACE to capture: saves a PNG with the 4×4 T_base_ee matrix embedded as a
     PNG text chunk (key "ee_pose") plus a companion JSON file.
  4. Collect ≥ 15 diverse poses (vary orientation and height).
  5. Run solve_hand_eye.py (or use OpenCV's calibrateHandEye) on the saved data.

Output directory layout:
  <OUTPUT_DIR>/
      calib_0001.png    — captured frame with embedded pose metadata
      calib_0001.json   — same pose as plain JSON (convenient for post-processing)
      calib_0002.png / .json ...
      poses.npz         — accumulated R_gripper2base, t_gripper2base arrays (updated live)

Controls:
  t                    toggle compliance  (robot goes limp — move freely by hand;
                       press t again to lock position and return to normal mode)
  Arrow Up / Down      jog +X / -X  (robot forward/back)
  Arrow Left / Right   jog -Y / +Y  (robot left/right)
  [ / ]                jog -Z / +Z  (robot down/up)
  SPACE                capture current frame + EE pose  (torque must be ON)
  r / HOME             return to home position
  s                    print current EE position to terminal
  h                    toggle help overlay
  q / Esc              return home and quit
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ikpy.chain import Chain

try:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    _PIL = True
except ImportError:
    _PIL = False
    print("[warn] Pillow not found — PNG metadata embedding disabled; raw PNG saved instead.")
    print("       Install with:  pip install pillow")

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

ROLL_FIXED_DEG = -90.0

CAMERA_INDEX  = 1
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

# Where calibration images and JSON files are saved
OUTPUT_DIR = Path(r"C:\Users\plata\robots\lerobot\calibration\hand_eye_data_4")

# Home configuration (same as click_to_move.py)
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
HOME_DEG = np.array([-1.2308, -99.5604, 97.8901, 34.4615, -89.6264, 99.8606])

# Motion parameters (matched to click_to_move.py)
SMOOTH_DURATION_S  = 2.0
KP = 1.0
KI = 5.0
KD = 0.0
SETTLE_THRESHOLD_M = 0.003
SETTLE_MAX_ITER    = 60
D_ALPHA            = 0.3

JOG_XY_M = 0.02   # metres per arrow-key press
JOG_Z_M  = 0.02   # metres per [ / ] press

# Windows virtual-key codes from cv2.waitKeyEx
_KEY_UP    = 2490368
_KEY_DOWN  = 2621440
_KEY_LEFT  = 2424832
_KEY_RIGHT = 2555904
_KEY_HOME  = 2359296
_KEY_SPACE = 32

# ─── Kinematics ───────────────────────────────────────────────────────────────

class SO101Kinematics:
    def __init__(self, urdf_path: str):
        _probe = Chain.from_urdf_file(urdf_path)
        self._link_names = [lnk.name for lnk in _probe.links]

        _active = [j for j in ARM_JOINTS if j != "wrist_roll"]
        mask = [name in _active for name in self._link_names]

        self.chain = Chain.from_urdf_file(urdf_path, active_links_mask=mask)
        self._arm_idx        = [self._link_names.index(n) for n in ARM_JOINTS]
        self._wrist_roll_idx = self._link_names.index("wrist_roll")

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
        """6-element degree array → 4×4 base-to-EE transform."""
        return self.chain.forward_kinematics(self._to_ikpy(joint_pos_deg[:5]))

    def inverse_kinematics(self, current_deg: np.ndarray, target_T: np.ndarray) -> np.ndarray:
        initial = self._clip_to_bounds(self._to_ikpy(current_deg[:5]))
        initial[self._wrist_roll_idx] = np.deg2rad(ROLL_FIXED_DEG)
        result = self.chain.inverse_kinematics(
            target_position=target_T[:3, 3],
            initial_position=initial,
        )
        out = np.array(current_deg, dtype=float)
        out[:5] = self._from_ikpy(result)
        out[4]  = ROLL_FIXED_DEG
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


# ─── Motion ───────────────────────────────────────────────────────────────────

def settle(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    target_T: np.ndarray,
    stop_event: threading.Event,
) -> str:
    integral   = np.zeros(3)
    prev_error = np.zeros(3)
    deriv_filt = np.zeros(3)
    dt         = 1.0 / FPS

    for _ in range(SETTLE_MAX_ITER):
        if stop_event.is_set():
            return "cancelled"

        t0  = time.perf_counter()
        obs = robot.get_observation()
        q   = _joints_from_obs(obs)
        p   = kin.forward_kinematics(q)[:3, 3]

        error = target_pos - p
        if float(np.linalg.norm(error)) < SETTLE_THRESHOLD_M:
            return "done"

        integral  += error * dt
        raw_d      = (error - prev_error) / dt
        deriv_filt = D_ALPHA * raw_d + (1 - D_ALPHA) * deriv_filt
        correction = KP * error + KI * integral + KD * deriv_filt
        prev_error = error

        cn = float(np.linalg.norm(correction))
        if cn > 0.05:
            correction *= 0.05 / cn

        T_cmd = target_T.copy()
        T_cmd[:3, 3] = p + correction
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
    duration_s: float = SMOOTH_DURATION_S,
) -> str:
    obs     = robot.get_observation()
    q       = _joints_from_obs(obs)
    T_start = kin.forward_kinematics(q)
    p_start = T_start[:3, 3].copy()
    gripper = obs["gripper.pos"]

    target_pos = _clamp_target(target_pos, p_start)
    if float(np.linalg.norm(target_pos - p_start)) < 1e-4:
        return "done"

    T_target = T_start.copy()
    T_target[:3, 3] = target_pos

    n_steps = max(int(duration_s * FPS), 1)
    for i in range(1, n_steps + 1):
        if stop_event.is_set():
            return "cancelled"
        t0    = time.perf_counter()
        alpha = i / n_steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        T_wp = T_start.copy()
        T_wp[:3, 3] = (1.0 - alpha) * p_start + alpha * target_pos
        q = kin.inverse_kinematics(q, T_wp)

        action = {f"{n}.pos": float(q[j]) for j, n in enumerate(MOTOR_NAMES) if n != "gripper"}
        action["gripper.pos"] = gripper
        robot.send_action(action)
        time.sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    return settle(robot, kin, target_pos, T_target, stop_event)


# ─── Capture ──────────────────────────────────────────────────────────────────

def _save_sample(
    frame: np.ndarray,
    T_base_ee: np.ndarray,
    output_dir: Path,
    index: int,
) -> Path:
    """
    Save frame as PNG with T_base_ee embedded in a PNG text chunk ("ee_pose").
    Also writes a sidecar .json with the same data.
    Appends R/t to poses.npz for batch loading.
    Returns path to the saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"calib_{index:04d}"

    R = T_base_ee[:3, :3]
    t = T_base_ee[:3,  3]

    pose_data = {
        "index":           index,
        "T_base_ee":       T_base_ee.tolist(),
        "R_gripper2base":  R.tolist(),
        "t_gripper2base":  t.tolist(),
    }

    # ── sidecar JSON ─────────────────────────────────────────────────────────
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(pose_data, indent=2))

    # ── PNG with embedded metadata ────────────────────────────────────────────
    png_path = output_dir / f"{stem}.png"
    if _PIL:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        info    = PngInfo()
        info.add_text("ee_pose", json.dumps(pose_data))
        pil_img.save(str(png_path), pnginfo=info)
    else:
        cv2.imwrite(str(png_path), frame)

    # ── accumulate poses.npz ─────────────────────────────────────────────────
    npz_path = output_dir / "poses.npz"
    if npz_path.exists():
        old = np.load(str(npz_path))
        R_all = list(old["R_gripper2base"])
        t_all = list(old["t_gripper2base"])
    else:
        R_all, t_all = [], []

    R_all.append(R)
    t_all.append(t)
    np.savez(str(npz_path), R_gripper2base=np.array(R_all), t_gripper2base=np.array(t_all))

    return png_path


# ─── GUI ──────────────────────────────────────────────────────────────────────

_HELP = [
    "t                    toggle compliance (limp / hold)",
    "Arrow Up / Down      jog +X / -X",
    "Arrow Left / Right   jog -Y / +Y",
    "[ / ]                jog -Z / +Z",
    "SPACE                capture (torque must be ON)",
    "r / HOME             return to home position",
    "s                    print EE position to terminal",
    "h                    toggle this help",
    "q / Esc              return home and quit",
]
_WIN = "SO-101  Hand-Eye Calibration"


class CalibrationApp:
    def __init__(
        self,
        robot: SO101Follower,
        kin: SO101Kinematics,
        cap: cv2.VideoCapture,
        home_pos: np.ndarray,
        output_dir: Path,
    ):
        self.robot      = robot
        self.kin        = kin
        self.cap        = cap
        self._home      = home_pos.copy()
        self._out_dir   = output_dir

        self._count     = 0
        self._status    = "Ready — press T to go compliant, jog with arrows/[ ], SPACE to capture"
        self._help      = False
        self._flash     = 0
        self._compliant = False   # True while torque is disabled

        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock   = threading.Lock()

    # ── compliance ────────────────────────────────────────────────────────────

    def _enable_torque(self) -> bool:
        """Freeze at current joint positions then re-enable torque. Returns True on success."""
        try:
            obs = self.robot.get_observation()
            # Write present positions as goal before enabling torque to prevent snap-back.
            action = {f"{n}.pos": obs[f"{n}.pos"] for n in MOTOR_NAMES}
            self.robot.send_action(action)
            self.robot.bus.enable_torque()
            self._compliant = False
            return True
        except Exception as exc:
            self._status = f"Failed to enable torque: {exc}"
            return False

    def _toggle_compliance(self) -> None:
        if self._compliant:
            if self._enable_torque():
                self._status = "Torque ON — position held. Press SPACE to capture."
        else:
            # Cancel any active move first.
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            self._stop.clear()
            try:
                self.robot.bus.disable_torque()
                self._compliant = True
                self._status = "COMPLIANT — move robot freely. Press T to lock."
            except Exception as exc:
                self._status = f"Failed to disable torque: {exc}"

    # ── motion ────────────────────────────────────────────────────────────────

    def _go_home(self) -> None:
        if self._compliant:
            if not self._enable_torque():
                return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._move_worker, args=(self._home, "Home"), daemon=True
        )
        self._thread.start()

    def _jog(self, dx: float = 0, dy: float = 0, dz: float = 0) -> None:
        if self._compliant:
            if not self._enable_torque():
                return
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

    def _move_worker(self, target: np.ndarray, label: str = "Target") -> None:
        tgt = f"({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:.3f}) m"
        self._status = f"{label} → {tgt} …"
        result = smooth_move(self.robot, self.kin, target, self._stop)
        self._status = {
            "done":      f"Arrived at {tgt}  — press SPACE to capture",
            "cancelled": "Move cancelled",
            "max_iter":  f"Max settle iters near {tgt}",
        }.get(result, result)

    def _is_moving(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── capture ───────────────────────────────────────────────────────────────

    def _capture(self, frame: np.ndarray) -> None:
        if self._compliant:
            self._status = "Torque is OFF — press T to lock position first, then SPACE to capture"
            return
        if self._is_moving():
            self._status = "Still moving — wait for robot to settle before capturing"
            return

        obs      = self.robot.get_observation()
        q        = _joints_from_obs(obs)
        T_base_ee = self.kin.forward_kinematics(q)

        self._count += 1
        try:
            path = _save_sample(frame.copy(), T_base_ee, self._out_dir, self._count)
        except Exception as exc:
            self._status = f"Save failed: {exc}"
            self._count -= 1
            return

        R = T_base_ee[:3, :3]
        t = T_base_ee[:3,  3]
        print(f"\n[{self._count:04d}] Captured → {path.name}")
        print(f"  t = ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m")
        print(f"  R = {np.array2string(R[0], precision=4, sign='+')}")
        print(f"      {np.array2string(R[1], precision=4, sign='+')}")
        print(f"      {np.array2string(R[2], precision=4, sign='+')}")

        self._status = f"Captured #{self._count}  → {path.name}  (total: {self._count})"
        self._flash  = 6   # show white flash for ~6 frames

    # ── overlay ───────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # compliance banner — full-width orange bar at top
        if self._compliant:
            cv2.rectangle(frame, (0, 0), (w, 38), (0, 100, 220), -1)
            cv2.putText(frame, "COMPLIANT  — move robot freely — press T to lock",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # capture flash — briefly brighten frame
        if self._flash > 0:
            alpha = self._flash / 6.0
            frame = cv2.addWeighted(frame, 1.0 - alpha * 0.5,
                                    np.full_like(frame, 255), alpha * 0.5, 0)
            self._flash -= 1

        # status bar
        bar_h = 58
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, self._status, (8, h - bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        info = (f"Captured: {self._count}  |  "
                f"t=compliant  arrows=jog XY  [/]=jog Z  SPACE=capture  h=help  q=quit")
        cv2.putText(frame, info, (8, h - bar_h + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (160, 220, 160), 1, cv2.LINE_AA)

        # capture-count badge — shift below compliance banner if active
        badge_y = 64 if self._compliant else 28
        badge   = f"#{self._count}"
        cv2.putText(frame, badge, (8, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 220, 80) if self._count > 0 else (100, 100, 100),
                    2, cv2.LINE_AA)

        # help overlay
        if self._help:
            for i, line in enumerate(_HELP):
                cv2.putText(frame, line, (10, 56 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 200), 1, cv2.LINE_AA)

        return frame

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

        print(f"{'─'*60}")
        print("  SO-101 Hand-Eye Calibration data collector")
        print(f"  Camera {CAMERA_INDEX}  {CAMERA_WIDTH}×{CAMERA_HEIGHT}")
        print(f"  Output: {self._out_dir}")
        print("  Jog the robot, press SPACE to capture.  [h] for help.")
        print(f"{'─'*60}\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "No camera signal", (80, CAMERA_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 200), 2)

            self._draw(frame)
            cv2.imshow(_WIN, frame)

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue

            if   key == _KEY_UP:    self._jog(dx=+JOG_XY_M)
            elif key == _KEY_DOWN:  self._jog(dx=-JOG_XY_M)
            elif key == _KEY_LEFT:  self._jog(dy=-JOG_XY_M)
            elif key == _KEY_RIGHT: self._jog(dy=+JOG_XY_M)
            elif key == _KEY_SPACE:
                ret2, cap_frame = self.cap.read()
                self._capture(cap_frame if ret2 else frame)
            elif key == _KEY_HOME:
                self._go_home()
            elif key & 0xFF == ord("t"):
                self._toggle_compliance()
            elif key & 0xFF == ord("["):  self._jog(dz=-JOG_Z_M)
            elif key & 0xFF == ord("]"):  self._jog(dz=+JOG_Z_M)
            elif key & 0xFF in (ord("q"), 27):
                break
            elif key & 0xFF == ord("r"):
                self._go_home()
            elif key & 0xFF == ord("h"):
                self._help = not self._help
            elif key & 0xFF == ord("s"):
                obs = self.robot.get_observation()
                q   = _joints_from_obs(obs)
                T   = self.kin.forward_kinematics(q)
                p   = T[:3, 3]
                print(f"EE  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}")

        # Always re-enable torque before disconnecting.
        if self._compliant:
            self._enable_torque()

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        print(f"\nSession ended.  Total captures: {self._count}")
        if self._count > 0:
            print(f"Data saved to: {self._out_dir}")


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

    home = kin.forward_kinematics(HOME_DEG)[:3, 3].copy()
    print(f"Home: x={home[0]:+.4f}  y={home[1]:+.4f}  z={home[2]:+.4f}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {CAMERA_INDEX} opened at {actual_w}×{actual_h}")
    else:
        print(f"[warn] Could not open camera index {CAMERA_INDEX}")

    app = CalibrationApp(robot, kin, cap, home_pos=home, output_dir=OUTPUT_DIR)
    try:
        app.run()
    finally:
        print("\nReturning to home position …")
        stop_dummy = threading.Event()
        smooth_move(robot, kin, home, stop_dummy)
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
