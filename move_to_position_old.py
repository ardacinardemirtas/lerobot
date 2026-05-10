#!/usr/bin/env python
"""
Interactive end-effector controller for SO-101.
Uses ikpy for FK/IK — pure Python, works on Windows natively.

Install dependency:  pip install ikpy

Commands (at the prompt):
  status / s              - print current EE position
  go <x|y|z> <meters>    - move relative to current position along one axis
  move <x> <y> <z>       - move to absolute position (meters)
  quit / q                - exit

Examples:
  > go z 0.05       # 5 cm upward
  > go x -0.03      # 3 cm in -x
  > move 0.15 0 0.10
"""

import re
import time
from pathlib import Path

import numpy as np
from ikpy.chain import Chain

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ── Configuration ─────────────────────────────────────────────────────────────
PORT            = "COM5"   # change to your port
ROBOT_ID        = "my_awesome_follower_arm"
CALIBRATION_DIR = Path(r"C:\Users\plata\robots")
URDF_PATH       = r"C:\Users\plata\robots\lerobot\calibration\so101_new_calib.urdf"
FPS             = 30

# The 5 revolute joints that control EE position (gripper open/close is separate)
ARM_JOINTS  = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]

ROLL_FIXED_DEG = -90.0   # wrist roll held at this angle; change to reorient gripper

# Safe workspace bounds in meters
WS_MIN      = np.array([-0.35, -0.35,  -0.10])
WS_MAX      = np.array([ 0.35,  0.35,  0.50])
MAX_MOVE_M  = 0.30   # maximum single-move distance (safety clamp)
# ─────────────────────────────────────────────────────────────────────────────


class SO101Kinematics:
    """
    Wraps ikpy for FK and IK on the SO-101 arm.

    Only the 5 arm joints are active; the gripper joint is kept passive so it
    does not interfere with the solver and is handled by the caller.
    """

    def __init__(self, urdf_path: str):
        # Load full chain so we can discover link names and build the mask
        _probe = Chain.from_urdf_file(urdf_path)
        self._link_names = [lnk.name for lnk in _probe.links]

        # Active = 4 arm joints; wrist_roll is passive (held at ROLL_FIXED_DEG)
        _active = [j for j in ARM_JOINTS if j != "wrist_roll"]
        mask = [name in _active for name in self._link_names]

        self.chain = Chain.from_urdf_file(urdf_path, active_links_mask=mask)
        self._arm_idx = [self._link_names.index(n) for n in ARM_JOINTS]
        self._wrist_roll_link_idx = self._link_names.index("wrist_roll")

        print("Kinematic chain:")
        for i, name in enumerate(self._link_names):
            tag = " ← active" if mask[i] else ""
            print(f"  [{i:2d}] {name}{tag}")
        print()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _to_ikpy(self, arm_deg: np.ndarray) -> np.ndarray:
        """5 arm joint degrees → full ikpy angle vector (radians, zeros elsewhere)."""
        q = np.zeros(len(self._link_names))
        for ikpy_i, deg in zip(self._arm_idx, arm_deg):
            q[ikpy_i] = np.deg2rad(deg)
        return q

    def _from_ikpy(self, q: np.ndarray) -> np.ndarray:
        """Full ikpy angle vector → 5 arm joint degrees."""
        return np.array([np.rad2deg(q[i]) for i in self._arm_idx])

    def _clip_to_bounds(self, q: np.ndarray) -> np.ndarray:
        """Clip full ikpy angle vector to each joint's URDF limits."""
        q = q.copy()
        for i, link in enumerate(self.chain.links):
            if link.bounds is not None:
                lo, hi = link.bounds
                if lo is not None:
                    q[i] = max(q[i], lo)
                if hi is not None:
                    q[i] = min(q[i], hi)
        return q

    # ── public API ────────────────────────────────────────────────────────────

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        joint_pos_deg: 6-element array (5 arm joints + gripper) in degrees.
        Returns 4×4 world-to-EE transformation matrix.
        """
        return self.chain.forward_kinematics(self._to_ikpy(joint_pos_deg[:5]))

    def inverse_kinematics(
        self, current_deg: np.ndarray, target_T: np.ndarray
    ) -> np.ndarray:
        """
        current_deg: 6-element array (used as IK warm-start and for gripper passthrough).
        target_T:    4×4 desired EE pose.
        Returns new 6-element degree array (gripper value preserved from current_deg).
        """
        # Clip initial guess to URDF joint limits — motor degree values can
        # sit just outside the limits due to calibration tolerances.
        initial = self._clip_to_bounds(self._to_ikpy(current_deg[:5]))
        # Fix wrist_roll in the initial vector — optimizer leaves passive joints untouched
        initial[self._wrist_roll_link_idx] = np.deg2rad(ROLL_FIXED_DEG)

        result = self.chain.inverse_kinematics(
            target_position=target_T[:3, 3],
            initial_position=initial,
        )
        out = np.array(current_deg, dtype=float)
        out[:5] = self._from_ikpy(result)
        out[4] = ROLL_FIXED_DEG   # enforce fixed roll explicitly
        return out


# ── Helpers ───────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def _parse_num(token: str) -> float:
    """Extract a float from a token like 'x=+0.24m' or '-0.0135'."""
    m = _NUM_RE.search(token)
    if m is None:
        raise ValueError(token)
    return float(m.group())


def _joints_from_obs(obs: dict) -> np.ndarray:
    return np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=float)


def _print_ee(T: np.ndarray, prefix: str = ""):
    p = T[:3, 3]
    R = T[:3, :3]
    print(f"{prefix}EE → x={p[0]:+.4f}m  y={p[1]:+.4f}m  z={p[2]:+.4f}m")
    print(f"{prefix}R  = {np.array2string(R[0], precision=4, sign='+')}")
    print(f"{prefix}     {np.array2string(R[1], precision=4, sign='+')}")
    print(f"{prefix}     {np.array2string(R[2], precision=4, sign='+')}")


def _clamp_target(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    target = np.clip(target, WS_MIN, WS_MAX)
    dist = float(np.linalg.norm(target - current))
    if dist > MAX_MOVE_M:
        target = current + (target - current) * (MAX_MOVE_M / dist)
        print(f"  [safety] clamped to {MAX_MOVE_M:.2f}m from start")
    return target


# ── Motion ────────────────────────────────────────────────────────────────────

# PID gains for the Cartesian settle loop.
# Start with KI=0, KD=0 and raise KP until sustained oscillation for Z-N tuning.
KP = 1.0    # proportional
KI = 5.0    # integral — eliminates steady-state error from friction/deadband
KD = 0.0    # derivative — damps oscillation

SETTLE_THRESHOLD_M = 0.003   # stop when FK error < 3 mm
SETTLE_MAX_ITER    = 60      # safety cap (~2 s at 30 Hz)
D_ALPHA            = 0.3     # derivative low-pass coefficient (0=frozen, 1=raw)


def settle(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    target_T: np.ndarray,
):
    """
    PID settle loop in Cartesian space.

    Runs after smooth_move to drive the FK-reported position error to zero.
    Error = target_pos − FK(actual_joint_angles)[:3,3]
    The PID output shifts the IK target each step until error < SETTLE_THRESHOLD_M.

    Also performs Z-N peak detection: prints Tu and suggested gains when
    sustained oscillations are detected (useful while tuning KP).
    """
    integral   = np.zeros(3)
    prev_error = np.zeros(3)
    deriv_filt = np.zeros(3)
    dt         = 1.0 / FPS
    t_start    = time.perf_counter()

    dist_buf:   list[float] = []
    t_buf:      list[float] = []
    peak_times: list[float] = []

    print(f"  {'t(s)':>6}  {'|err|mm':>8}  {'x mm':>8}  {'y mm':>8}  {'z mm':>8}")

    for _ in range(SETTLE_MAX_ITER):
        t0 = time.perf_counter()

        obs    = robot.get_observation()
        q      = _joints_from_obs(obs)
        T_curr = kin.forward_kinematics(q)
        p_curr = T_curr[:3, 3]

        error   = target_pos - p_curr
        dist    = float(np.linalg.norm(error))
        elapsed = t0 - t_start

        print(f"  {elapsed:6.3f}  {dist*1000:8.2f}"
              f"  {error[0]*1000:+8.2f}  {error[1]*1000:+8.2f}  {error[2]*1000:+8.2f}")

        # 3-point peak detection on scalar |err| for Z-N Tu estimation
        dist_buf.append(dist)
        t_buf.append(elapsed)
        if len(dist_buf) >= 3:
            d0, d1, d2 = dist_buf[-3], dist_buf[-2], dist_buf[-1]
            if d1 > d0 and d1 > d2 and d1 > SETTLE_THRESHOLD_M * 3:
                peak_times.append(t_buf[-2])

        if dist < SETTLE_THRESHOLD_M:
            print(f"  ✓ settled  t={elapsed:.3f}s  final={dist*1000:.1f}mm  KP={KP}")
            break

        # PID — derivative is low-pass filtered to suppress sensor noise
        integral  += error * dt
        raw_deriv  = (error - prev_error) / dt
        deriv_filt = D_ALPHA * raw_deriv + (1 - D_ALPHA) * deriv_filt
        correction = KP * error + KI * integral + KD * deriv_filt
        prev_error = error

        # Clamp correction magnitude to prevent wild IK jumps
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
    else:
        obs       = robot.get_observation()
        T_final   = kin.forward_kinematics(_joints_from_obs(obs))
        remaining = float(np.linalg.norm(target_pos - T_final[:3, 3]))
        print(f"  ✗ max iters  residual={remaining*1000:.1f}mm  KP={KP}")

    # Z-N Tu from peak-to-peak intervals (need ≥ 2 peaks = one full cycle)
    if len(peak_times) >= 2:
        periods = [peak_times[i+1] - peak_times[i] for i in range(len(peak_times)-1)]
        Tu_est  = float(np.median(periods))
        Ku      = KP
        print(f"\n  Peaks: {len(peak_times)}  Tu ≈ {Tu_est:.3f}s  Ku = {Ku}")
        print(f"  Z-N no-overshoot:   KP={0.20*Ku:.3f}  KI={0.40*Ku/Tu_est:.3f}  KD={0.066*Ku*Tu_est:.4f}")
        print(f"  Z-N some-overshoot: KP={0.33*Ku:.3f}  KI={0.66*Ku/Tu_est:.3f}  KD={0.110*Ku*Tu_est:.4f}")
    elif len(peak_times) == 1:
        print("  Only 1 peak — need sustained oscillation. Raise KP slightly.")
    else:
        trend = "converging — raise KP" if (dist_buf and dist_buf[-1] < dist_buf[0]) else "diverging — lower KP"
        print(f"  No peaks detected ({trend})")


def smooth_move(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    duration_s: float = 2.0,
):
    """
    Smoothstep Cartesian interpolation to target_pos, then PID settle.
    Orientation is held constant throughout.
    """
    obs    = robot.get_observation()
    q      = _joints_from_obs(obs)
    T_start = kin.forward_kinematics(q)
    p_start = T_start[:3, 3].copy()
    gripper_pos = obs["gripper.pos"]

    target_pos = _clamp_target(target_pos, p_start)
    dist = float(np.linalg.norm(target_pos - p_start))
    if dist < 1e-4:
        print("  Already at target.")
        return

    T_target = T_start.copy()
    T_target[:3, 3] = target_pos

    n_steps = max(int(duration_s * FPS), 1)
    print(f"  Moving {dist * 100:.1f} cm in {duration_s:.1f}s …")

    for i in range(1, n_steps + 1):
        t0 = time.perf_counter()

        alpha = i / n_steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep easing

        T_wp = T_start.copy()
        T_wp[:3, 3] = (1.0 - alpha) * p_start + alpha * target_pos

        q = kin.inverse_kinematics(q, T_wp)  # warm-start from previous q

        action = {f"{n}.pos": float(q[j]) for j, n in enumerate(MOTOR_NAMES) if n != "gripper"}
        action["gripper.pos"] = gripper_pos
        robot.send_action(action)

        time.sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    # PID settle to drive residual FK error to zero
    settle(robot, kin, target_pos, T_target)


# ── REPL ──────────────────────────────────────────────────────────────────────

HELP = """\
  status / s              print current EE position
  go <x|y|z> <meters>    relative move along one axis
  move <x> <y> <z>       absolute move to position (meters)
  quit / q                disconnect and exit"""


def run_repl(robot: SO101Follower, kin: SO101Kinematics):
    print("SO-101 EE Controller  —  type 'help' for commands\n")

    while True:
        obs = robot.get_observation()
        q = _joints_from_obs(obs)
        T = kin.forward_kinematics(q)
        _print_ee(T)

        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        tokens = raw.lower().split()
        cmd = tokens[0]

        if cmd in ("q", "quit", "exit"):
            break

        elif cmd in ("s", "status"):
            continue

        elif cmd == "help":
            print(HELP)

        elif cmd == "go":
            if len(tokens) != 3:
                print("  Usage: go <x|y|z> <delta_meters>")
                continue
            axis_map = {"x": 0, "y": 1, "z": 2}
            if tokens[1] not in axis_map:
                print("  Axis must be x, y, or z")
                continue
            try:
                delta = _parse_num(tokens[2])
            except ValueError:
                print("  Delta must be a number")
                continue

            target = T[:3, 3].copy()
            target[axis_map[tokens[1]]] += delta
            smooth_move(robot, kin, target)

        elif cmd == "move":
            # Accept plain numbers, labels (x=0.1), or copy-pasted EE output (x=+0.24m)
            nums = []
            for t in tokens[1:]:
                try:
                    nums.append(_parse_num(t))
                except ValueError:
                    pass
            if len(nums) != 3:
                print("  Usage: move <x> <y> <z>  (e.g. move 0.15 0 0.10)")
                continue

            target = np.array(nums)
            smooth_move(robot, kin, target)

        else:
            print(f"  Unknown command '{cmd}'. Type 'help'.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    kin = SO101Kinematics(URDF_PATH)   # prints chain on startup, no robot needed yet

    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        use_degrees=True,
    )
    robot = SO101Follower(config)
    robot.connect()

    try:
        run_repl(robot, kin)
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
