#!/usr/bin/env python
"""
Interactive end-effector controller for SO-101.
Uses ikpy for FK only — IK replaced by Jacobian pseudoinverse controller.

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

# Safe workspace bounds in meters
WS_MIN      = np.array([-0.35, -0.35,  0.00])
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

        # Active = only the 5 arm joints; everything else (base, gripper, tip) passive
        mask = [name in ARM_JOINTS for name in self._link_names]

        self.chain = Chain.from_urdf_file(urdf_path, active_links_mask=mask)
        self._arm_idx = [self._link_names.index(n) for n in ARM_JOINTS]

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

        result = self.chain.inverse_kinematics(
            target_position=target_T[:3, 3],
            target_orientation=target_T[:3, :3],
            orientation_mode="all",
            initial_position=initial,
        )
        out = np.array(current_deg, dtype=float)
        out[:5] = self._from_ikpy(result)
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
    R = T[:3, :3]
    t = T[:3, 3]
    print(f"{prefix}EE → x={t[0]:+.4f}m  y={t[1]:+.4f}m  z={t[2]:+.4f}m")
    print(f"{prefix}t = {np.array2string(t, precision=6, sign='+')}")
    print(f"{prefix}R = {np.array2string(R, precision=6, sign='+')}")


def _clamp_target(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    target = np.clip(target, WS_MIN, WS_MAX)
    dist = float(np.linalg.norm(target - current))
    if dist > MAX_MOVE_M:
        target = current + (target - current) * (MAX_MOVE_M / dist)
        print(f"  [safety] clamped to {MAX_MOVE_M:.2f}m from start")
    return target


# ── Motion ────────────────────────────────────────────────────────────────────

POS_THRESHOLD_M = 0.003   # converged when position error < 3 mm
ORI_THRESHOLD   = 0.05    # converged when orientation error < ~3 deg
JAC_MAX_ITER    = 300     # safety cap (~10 s at 30 Hz)
JAC_ALPHA       = 0.8     # step-size gain — reduce if oscillating
JAC_LAMBDA      = 0.01    # DLS damping — increase near singularities
JAC_W_ORI       = 0.3     # orientation weight (rad vs m unit scale)
STALL_WINDOW    = 10      # steps to look back for stall detection (~0.33 s)
STALL_MIN_M     = 0.0005  # stall if EE moved less than 0.5 mm over the window

# PID gains applied to position error fed into the Jacobian step.
# Integral term eliminates steady-state offset; derivative damps oscillation.
PID_KP          = 1.0     # proportional — keep at 1.0 (JAC_ALPHA handles scaling)
PID_KI          = 5.0     # integral     — increase to kill residual offset faster
PID_KD          = 0.05    # derivative   — increase to damp overshoot
PID_I_CLAMP     = 0.05    # integral windup clamp (meters)


def _rot_error(R_target: np.ndarray, R_curr: np.ndarray) -> np.ndarray:
    """Axis-angle error vector: how much R_curr must rotate to reach R_target."""
    R_err = R_target @ R_curr.T
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    if abs(angle) < 1e-8:
        return np.zeros(3)
    return (angle / (2.0 * np.sin(angle))) * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def jacobian_move(
    robot: SO101Follower,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    pid_enabled: bool = False,
):
    """
    Damped least-squares Jacobian controller.

    Moves the end-effector to target_pos while holding the orientation fixed
    at whatever it is when the move starts.  Uses only FK — no IK solver.

    Each step:
      1. Read actual joint angles from hardware.
      2. FK → current EE pose.
      3. Compute 6D error (3 position + 3 orientation).
      4. Build 6×5 numerical Jacobian via finite differences (5 FK calls).
      5. Δq = Jᵀ(JJᵀ + λI)⁻¹ · error   (damped pseudoinverse)
      6. Apply step, clamp to URDF limits, send.
    """
    obs = robot.get_observation()
    q_deg = _joints_from_obs(obs)
    q_rad = kin._to_ikpy(q_deg[:5])
    T_start  = kin.chain.forward_kinematics(q_rad)
    R_target = T_start[:3, :3].copy()

    target_pos = _clamp_target(target_pos, T_start[:3, 3])
    dist0 = float(np.linalg.norm(target_pos - T_start[:3, 3]))
    if dist0 < 1e-4:
        print("  Already at target.")
        return

    pid_tag = " [PID]" if pid_enabled else ""
    print(f"  Moving {dist0 * 100:.1f} cm …{pid_tag}")
    print(f"  {'t(s)':>6}  {'pos mm':>8}  {'ori deg':>8}")

    dt    = 1.0 / FPS
    dq_fd = 1e-4   # finite-difference step size
    t_start = time.perf_counter()
    p_history: list[np.ndarray] = []

    pid_integral  = np.zeros(3)
    pid_prev_err  = np.zeros(3)

    for _ in range(JAC_MAX_ITER):
        t0 = time.perf_counter()

        obs   = robot.get_observation()
        q_deg = _joints_from_obs(obs)
        q_rad = kin._to_ikpy(q_deg[:5])
        T_curr = kin.chain.forward_kinematics(q_rad)

        pos_err  = target_pos - T_curr[:3, 3]
        rot_err  = _rot_error(R_target, T_curr[:3, :3])
        pos_dist = float(np.linalg.norm(pos_err))
        ori_dist = float(np.linalg.norm(rot_err))
        elapsed  = time.perf_counter() - t_start

        print(f"  {elapsed:6.3f}  {pos_dist*1000:8.2f}  {np.degrees(ori_dist):8.2f}")

        if pid_enabled:
            pid_integral = np.clip(pid_integral + pos_err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            pid_deriv    = (pos_err - pid_prev_err) / dt
            pos_err      = PID_KP * pos_err + PID_KI * pid_integral + PID_KD * pid_deriv
            pid_prev_err = pos_err.copy()

        ori_locked = float(np.linalg.norm(rot_err[:2]))
        if pos_dist < POS_THRESHOLD_M and ori_locked < ORI_THRESHOLD:
            print(f"  ✓ done  t={elapsed:.3f}s  pos={pos_dist*1000:.1f}mm  ori={np.degrees(ori_locked):.1f}°")
            break

        # Stall detection: if the EE barely moved over the last STALL_WINDOW steps, give up
        p_history.append(T_curr[:3, 3].copy())
        if len(p_history) > STALL_WINDOW:
            p_history.pop(0)
        if len(p_history) == STALL_WINDOW:
            window_travel = float(np.linalg.norm(p_history[-1] - p_history[0]))
            if window_travel < STALL_MIN_M:
                print(f"  ✗ stalled  travel={window_travel*1000:.2f}mm over {STALL_WINDOW} steps"
                      f"  pos={pos_dist*1000:.1f}mm remaining")
                break

        # 5×5 numerical Jacobian: 3 position + 2 orientation (roll free)
        J = np.zeros((5, 5))
        for i, idx in enumerate(kin._arm_idx):
            q_plus = q_rad.copy()
            q_plus[idx] += dq_fd
            T_plus = kin.chain.forward_kinematics(q_plus)

            J[:3, i] = (T_plus[:3, 3] - T_curr[:3, 3]) / dq_fd

            R_diff = T_plus[:3, :3] @ T_curr[:3, :3].T
            tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            a  = np.arccos(tr)
            if abs(a) < 1e-8:
                J[3:, i] = np.zeros(2)
            else:
                ax = (a / (2.0 * np.sin(a) * dq_fd)) * np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1],
                ])
                J[3:, i] = ax[:2]  # pitch + yaw only, roll free

        error_5d = np.concatenate([pos_err, JAC_W_ORI * rot_err[:2]])
        dq_arm   = J.T @ np.linalg.solve(J @ J.T + JAC_LAMBDA * np.eye(5), error_5d)

        # Apply step and clamp to URDF joint limits
        q_new = q_rad.copy()
        for i, idx in enumerate(kin._arm_idx):
            q_new[idx] += JAC_ALPHA * dq_arm[i]
            link = kin.chain.links[idx]
            if link.bounds is not None:
                lo, hi = link.bounds
                if lo is not None:
                    q_new[idx] = max(q_new[idx], lo)
                if hi is not None:
                    q_new[idx] = min(q_new[idx], hi)

        q_deg_new = q_deg.copy()
        q_deg_new[:5] = kin._from_ikpy(q_new)

        action = {f"{n}.pos": float(q_deg_new[j]) for j, n in enumerate(MOTOR_NAMES) if n != "gripper"}
        action["gripper.pos"] = obs["gripper.pos"]
        robot.send_action(action)

        time.sleep(max(dt - (time.perf_counter() - t0), 0.0))
    else:
        obs     = robot.get_observation()
        T_final = kin.chain.forward_kinematics(kin._to_ikpy(_joints_from_obs(obs)[:5]))
        rem_pos = float(np.linalg.norm(target_pos - T_final[:3, 3]))
        rem_ori = float(np.linalg.norm(_rot_error(R_target, T_final[:3, :3])))
        print(f"  ✗ max iters  pos={rem_pos*1000:.1f}mm  ori={np.degrees(rem_ori):.1f}°")


# ── REPL ──────────────────────────────────────────────────────────────────────

HELP = """\
  status / s              print current EE position
  go <x|y|z> <meters>    relative move along one axis
  move <x> <y> <z>       absolute move to position (meters)
  pid                     toggle PID position correction on/off
  quit / q                disconnect and exit"""


def run_repl(robot: SO101Follower, kin: SO101Kinematics):
    print("SO-101 EE Controller  —  type 'help' for commands\n")

    obs = robot.get_observation()
    start_pos = kin.forward_kinematics(_joints_from_obs(obs))[:3, 3].copy()

    pid_enabled = False

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
            print("  Returning to start position …")
            jacobian_move(robot, kin, start_pos, pid_enabled)
            break

        elif cmd in ("s", "status"):
            continue

        elif cmd == "help":
            print(HELP)

        elif cmd == "pid":
            pid_enabled = not pid_enabled
            print(f"  PID {'enabled' if pid_enabled else 'disabled'}"
                  f"  (Kp={PID_KP}  Ki={PID_KI}  Kd={PID_KD})")

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
            jacobian_move(robot, kin, target, pid_enabled)

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
            jacobian_move(robot, kin, target, pid_enabled)

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
