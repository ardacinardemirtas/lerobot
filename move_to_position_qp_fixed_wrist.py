#!/usr/bin/env python
"""
Interactive end-effector controller for SO-101.
Uses minimum-jerk Cartesian reference + box-constrained QP differential IK.
No external IK library required — reads URDF directly.

Both wrist_roll and wrist_flex are held constant (not IK-controlled):
  - wrist_roll: FIXED_WRIST_ROLL_DEG = -90°
  - wrist_flex: midway between upper URDF limit and range centre (read from URDF at startup)

Only shoulder_pan, shoulder_lift, elbow_flex are IK-controlled.

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

import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ikpy.chain import Chain as _IKPyChain

# All 5 arm joints used by ikpy FK (wrist_roll included for correct EE position)
_IKPY_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# ── Configuration ─────────────────────────────────────────────────────────────
PORT            = "COM5"
ROBOT_ID        = "my_awesome_follower_arm"
CALIBRATION_DIR = Path(r"C:\Users\plata\robots")
URDF_PATH       = r"C:\Users\plata\robots\lerobot\calibration\so101_new_calib.urdf"
FPS             = 60

ARM_JOINTS  = ["shoulder_pan", "shoulder_lift", "elbow_flex"]  # wrist_flex and wrist_roll held fixed
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_INDEX = {name: i for i, name in enumerate(MOTOR_NAMES)}

FIXED_WRIST_ROLL_DEG = -90.0  # held constant; not IK-controlled
# FIXED_WRIST_FLEX_DEG is read from URDF at startup via kin.fixed_wrist_flex_deg

WS_MIN     = np.array([-0.35, -0.35, -0.10])
WS_MAX     = np.array([ 0.50,  0.50,  0.50])
MAX_MOVE_M = 0.30

# Controller gains (mirror of ControllerConfig defaults in move_to_position_new.py)
POSITION_GAIN          = 4.0
MAX_EE_SPEED_M_S       = 0.120
TERMINAL_EE_SPEED_M_S  = 0.025
MIN_DURATION_S         = 0.30
MAX_SETTLE_S           = 3.0
SETTLE_THRESHOLD_M     = 0.0020
STABLE_TICKS           = 5
SINGULAR_THRESHOLD     = 0.035
SINGULAR_DAMPING_GAIN  = 0.60
POSTURE_GAIN           = 0.18
JOINT_CENTER_GAIN      = 0.05
QP_TASK_WEIGHT         = 50.0
QP_DAMPING_WEIGHT      = 0.002
QP_ACCEL_WEIGHT        = 0.006
QP_POSTURE_WEIGHT      = 0.012
MAX_JOINT_SPEED_DEG_S  = 75.0
MAX_JOINT_ACCEL_DEG_S2 = 520.0
STOP_JOINT_SPEED_DEG_S = 1.0
COMMAND_DEADBAND_DEG   = 0.015
# ─────────────────────────────────────────────────────────────────────────────


# ── URDF kinematics ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def _parse_vec(text: str | None, default: Iterable[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.asarray(tuple(default), dtype=float)
    return np.asarray([float(x) for x in text.split()], dtype=float)


class SO101Kinematics:
    """FK and position Jacobian for SO-101 via direct URDF parsing (no ikpy)."""

    def __init__(self,
                 urdf_path: str | Path = URDF_PATH,
                 base_link: str = "base_link",
                 target_link: str = "gripper_frame_link") -> None:
        self.base_link     = base_link
        self.target_link   = target_link
        self.joints        = self._load_joints(Path(urdf_path))
        self.chain         = self._chain_to_target()
        self.active_joints = [n for n in ARM_JOINTS if n in self.joints]
        self.fixed_wrist_flex_deg = self._compute_fixed_wrist_flex()

        # ikpy chain for FK — includes wrist_roll so EE position is correct
        _probe = _IKPyChain.from_urdf_file(str(urdf_path))
        self._link_names   = [lnk.name for lnk in _probe.links]
        _active = [j for j in _IKPY_ARM_JOINTS if j != "wrist_roll"]
        mask = [name in _active for name in self._link_names]
        self._ikpy_chain   = _IKPyChain.from_urdf_file(str(urdf_path), active_links_mask=mask)
        self._ikpy_arm_idx = [self._link_names.index(n) for n in _IKPY_ARM_JOINTS]

    @staticmethod
    def _load_joints(path: Path) -> dict[str, JointSpec]:
        root = ET.parse(path).getroot()
        joints: dict[str, JointSpec] = {}
        for joint in root.findall("joint"):
            name       = joint.attrib["name"]
            joint_type = joint.attrib.get("type", "fixed")
            origin     = joint.find("origin")
            parent     = joint.find("parent")
            child      = joint.find("child")
            axis       = joint.find("axis")
            limit      = joint.find("limit")
            if parent is None or child is None:
                continue
            joints[name] = JointSpec(
                name=name,
                joint_type=joint_type,
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                xyz=_parse_vec(None if origin is None else origin.attrib.get("xyz")),
                rpy=_parse_vec(None if origin is None else origin.attrib.get("rpy")),
                axis=_parse_vec(None if axis is None else axis.attrib.get("xyz"), default=(0.0, 0.0, 1.0)),
                lower=None if limit is None or "lower" not in limit.attrib else float(limit.attrib["lower"]),
                upper=None if limit is None or "upper" not in limit.attrib else float(limit.attrib["upper"]),
            )
        return joints

    def _chain_to_target(self) -> list[JointSpec]:
        by_child = {joint.child: joint for joint in self.joints.values()}
        chain: list[JointSpec] = []
        link = self.target_link
        while link != self.base_link:
            if link not in by_child:
                raise ValueError(f"Could not find URDF chain from {self.base_link} to {self.target_link}")
            joint = by_child[link]
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return chain

    def _compute_fixed_wrist_flex(self) -> float:
        spec = self.joints.get("wrist_flex")
        if spec and spec.lower is not None and spec.upper is not None:
            lo  = math.degrees(spec.lower)
            hi  = math.degrees(spec.upper)
            mid = 0.5 * (lo + hi)
            return 0.5 * (hi + mid)  # midpoint between upper limit and range centre
        return 0.0

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Returns 4×4 EE transform using ikpy (includes wrist_roll for correct EE position)."""
        q_ikpy = np.zeros(len(self._link_names))
        for ikpy_idx, name in zip(self._ikpy_arm_idx, _IKPY_ARM_JOINTS):
            q_ikpy[ikpy_idx] = math.radians(float(joint_pos_deg[JOINT_INDEX[name]]))
        return self._ikpy_chain.forward_kinematics(q_ikpy)

    def bounds_deg(self, joint_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        names = joint_names or self.active_joints
        lower, upper = [], []
        for name in names:
            spec = self.joints[name]
            lower.append(-np.inf if spec.lower is None else math.degrees(spec.lower))
            upper.append( np.inf if spec.upper is None else math.degrees(spec.upper))
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def clip_joints(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        q = np.asarray(joint_pos_deg, dtype=float).copy()
        for name in ARM_JOINTS:
            spec = self.joints.get(name)
            if spec is None:
                continue
            idx = JOINT_INDEX[name]
            if spec.lower is not None:
                q[idx] = max(q[idx], math.degrees(spec.lower))
            if spec.upper is not None:
                q[idx] = min(q[idx], math.degrees(spec.upper))
        return q

    def position_jacobian(self,
                          joint_pos_deg: np.ndarray,
                          active_joints: list[str] | None = None,
                          eps_rad: float = 1e-4) -> np.ndarray:
        names   = active_joints or self.active_joints
        q       = np.asarray(joint_pos_deg, dtype=float)
        J       = np.zeros((3, len(names)), dtype=float)
        eps_deg = math.degrees(eps_rad)
        for col, name in enumerate(names):
            q_plus  = q.copy(); q_plus[JOINT_INDEX[name]]  += eps_deg
            q_minus = q.copy(); q_minus[JOINT_INDEX[name]] -= eps_deg
            p_plus  = self.forward_kinematics(q_plus)[:3, 3]
            p_minus = self.forward_kinematics(q_minus)[:3, 3]
            J[:, col] = (p_plus - p_minus) / (2.0 * eps_rad)
        return J


# ── Helpers ───────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def _parse_num(token: str) -> float:
    """Extract a float from a token like 'x=+0.24m' or '-0.0135'."""
    m = _NUM_RE.search(token)
    if m is None:
        raise ValueError(token)
    return float(m.group())

def _joints_from_obs(obs: dict) -> np.ndarray:
    return np.asarray([obs[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float)

def _print_ee(T: np.ndarray, prefix: str = "") -> None:
    p = T[:3, 3]
    print(f"{prefix}EE → x={p[0]:+.4f}m  y={p[1]:+.4f}m  z={p[2]:+.4f}m")

def _clamp_target(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    target = np.clip(target.astype(float), WS_MIN, WS_MAX)
    delta  = target - current
    dist   = float(np.linalg.norm(delta))
    if dist > MAX_MOVE_M:
        target = current + delta * (MAX_MOVE_M / dist)
        print(f"  [safety] clamped move to {MAX_MOVE_M:.2f}m")
    return target

def _clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > max_norm > 0.0:
        return vec * (max_norm / norm)
    return vec

def _minimum_jerk(tau: float) -> tuple[float, float]:
    tau    = float(np.clip(tau, 0.0, 1.0))
    s      = 10.0*tau**3 - 15.0*tau**4 + 6.0*tau**5
    ds_dtu = 30.0*tau**2 - 60.0*tau**3 + 30.0*tau**4
    return s, ds_dtu

def _active_q_rad(q_deg: np.ndarray, active_joints: list[str]) -> np.ndarray:
    return np.asarray([math.radians(float(q_deg[JOINT_INDEX[name]])) for name in active_joints], dtype=float)

def _write_active_q_rad(q_deg: np.ndarray, active_joints: list[str], q_active_rad: np.ndarray) -> np.ndarray:
    q = np.asarray(q_deg, dtype=float).copy()
    for value, name in zip(q_active_rad, active_joints):
        q[JOINT_INDEX[name]] = math.degrees(float(value))
    return q


# ── QP solver ─────────────────────────────────────────────────────────────────

def _solve_box_qp(H: np.ndarray, g: np.ndarray,
                  lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, int]:
    """Solve a small convex box-QP with an active-set clamp loop."""
    n     = int(g.size)
    H     = H + 1e-9 * np.eye(n)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    fixed = np.zeros(n, dtype=bool)
    x     = np.clip(np.zeros(n, dtype=float), lower, upper)

    for iteration in range(n + 1):
        free = ~fixed
        if np.any(free):
            rhs = -g[free]
            if np.any(fixed):
                rhs -= H[np.ix_(free, fixed)] @ x[fixed]
            try:
                x[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
            except np.linalg.LinAlgError:
                x[free] = np.linalg.lstsq(H[np.ix_(free, free)], rhs, rcond=None)[0]

        below     = x < lower
        above     = x > upper
        violation = np.maximum(lower - x, x - upper)
        if not np.any(below | above):
            return x, iteration + 1

        idx = int(np.argmax(violation))
        x[idx]     = lower[idx] if below[idx] else upper[idx]
        fixed[idx] = True

    return np.clip(x, lower, upper), n + 1


def _joint_limit_terms(
    kin: SO101Kinematics,
    q_deg: np.ndarray,
    active_joints: list[str],
    q_ref_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_rad      = _active_q_rad(q_deg, active_joints)
    q_ref_rad  = _active_q_rad(q_ref_deg, active_joints)
    lo_deg, hi_deg = kin.bounds_deg(active_joints)
    lo_rad     = np.deg2rad(lo_deg)
    hi_rad     = np.deg2rad(hi_deg)
    center_rad     = 0.5 * (lo_rad + hi_rad)
    half_range_rad = np.maximum(0.5 * (hi_rad - lo_rad), np.deg2rad(5.0))
    normalized     = np.clip((q_rad - center_rad) / half_range_rad, -1.5, 1.5)
    qdot_posture   = -POSTURE_GAIN * (q_rad - q_ref_rad)
    qdot_center    = -JOINT_CENTER_GAIN * normalized**3
    return q_rad, lo_rad, hi_rad, qdot_posture + qdot_center


def _joint_velocity_bounds(
    q_rad: np.ndarray,
    lo_rad: np.ndarray,
    hi_rad: np.ndarray,
    prev_qdot_rad_s: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    speed       = math.radians(MAX_JOINT_SPEED_DEG_S)
    accel_delta = math.radians(MAX_JOINT_ACCEL_DEG_S2)
    lower = np.maximum(-speed, (lo_rad - q_rad) / dt)
    upper = np.minimum( speed, (hi_rad - q_rad) / dt)
    lower = np.maximum(lower, prev_qdot_rad_s - accel_delta * dt)
    upper = np.minimum(upper, prev_qdot_rad_s + accel_delta * dt)
    return lower, upper


def _qp_qdot(
    kin: SO101Kinematics,
    q_deg: np.ndarray,
    v_task: np.ndarray,
    active_joints: list[str],
    q_ref_deg: np.ndarray,
    prev_qdot_rad_s: np.ndarray,
    dt: float,
) -> np.ndarray:
    J              = kin.position_jacobian(q_deg, active_joints)
    singular_values = np.linalg.svd(J, compute_uv=False)
    sigma_min      = float(singular_values[-1]) if len(singular_values) else 0.0
    singular_boost = max(0.0, SINGULAR_THRESHOLD - sigma_min)
    damping_weight = QP_DAMPING_WEIGHT + SINGULAR_DAMPING_GAIN * singular_boost * singular_boost

    q_rad, lo_rad, hi_rad, qdot_pref = _joint_limit_terms(kin, q_deg, active_joints, q_ref_deg)
    lower, upper = _joint_velocity_bounds(q_rad, lo_rad, hi_rad, prev_qdot_rad_s, dt)

    n = len(active_joints)
    H = (
        QP_TASK_WEIGHT   * (J.T @ J)
        + damping_weight * np.eye(n)
        + QP_ACCEL_WEIGHT   * np.eye(n)
        + QP_POSTURE_WEIGHT * np.eye(n)
    )
    g = (
        -QP_TASK_WEIGHT    * (J.T @ v_task)
        - QP_ACCEL_WEIGHT  * prev_qdot_rad_s
        - QP_POSTURE_WEIGHT * qdot_pref
    )
    qdot, _ = _solve_box_qp(H, g, lower, upper)
    return qdot


def _controller_step(
    kin: SO101Kinematics,
    q_deg: np.ndarray,
    q_ref_deg: np.ndarray,
    prev_qdot_rad_s: np.ndarray,
    p_start: np.ndarray,
    target_pos: np.ndarray,
    elapsed_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    dt          = 1.0 / FPS
    T_curr      = kin.forward_kinematics(q_deg)
    p_curr      = T_curr[:3, 3]
    tau         = elapsed_s / max(duration_s, 1e-6)
    s, ds_dtau  = _minimum_jerk(tau)
    final_phase = tau >= 1.0

    p_ref = p_start + s * (target_pos - p_start)
    if final_phase:
        p_ref = target_pos
        v_ff  = np.zeros(3)
    else:
        v_ff = (target_pos - p_start) * (ds_dtau / duration_s)

    error       = p_ref - p_curr
    final_error = target_pos - p_curr
    speed_cap   = TERMINAL_EE_SPEED_M_S if final_phase or np.linalg.norm(final_error) < 0.015 else MAX_EE_SPEED_M_S
    v_task      = _clamp_norm(v_ff + POSITION_GAIN * error, speed_cap)

    qdot  = _qp_qdot(kin, q_deg, v_task, kin.active_joints, q_ref_deg, prev_qdot_rad_s, dt)
    q_next = _write_active_q_rad(q_deg, kin.active_joints,
                                  _active_q_rad(q_deg, kin.active_joints) + qdot * dt)
    q_next = kin.clip_joints(q_next)
    return q_next, qdot, {
        "final_error_m": float(np.linalg.norm(final_error)),
        "sigma_min":     float(np.linalg.svd(kin.position_jacobian(q_deg, kin.active_joints), compute_uv=False)[-1]),
    }


# ── Motion ────────────────────────────────────────────────────────────────────

def smooth_move(robot, kin: SO101Kinematics, target_pos: np.ndarray) -> dict:
    """
    Minimum-jerk Cartesian path + QP differential IK.
    wrist_roll and wrist_flex are held fixed throughout; only the 3 proximal
    joints (shoulder_pan, shoulder_lift, elbow_flex) are IK-controlled.
    """
    dt    = 1.0 / FPS
    obs   = robot.get_observation()
    q_obs = kin.clip_joints(_joints_from_obs(obs))
    p_start = kin.forward_kinematics(q_obs)[:3, 3].copy()

    q_start = q_obs.copy()
    q_start[JOINT_INDEX["wrist_roll"]] = FIXED_WRIST_ROLL_DEG
    q_start[JOINT_INDEX["wrist_flex"]] = kin.fixed_wrist_flex_deg

    target   = _clamp_target(np.asarray(target_pos, dtype=float), p_start)
    distance = float(np.linalg.norm(target - p_start))
    if distance < 1e-4:
        print("  Already at target.")
        return {"success": True, "final_error_mm": 0.0, "elapsed_s": 0.0}

    duration = float(max(MIN_DURATION_S, 1.875 * distance / MAX_EE_SPEED_M_S))
    print(f"  wrist_flex fixed at {kin.fixed_wrist_flex_deg:.1f}°")
    print(f"  Moving {distance * 100.0:.1f} cm  reference time {duration:.2f}s")
    print(f"  {'t(s)':>6}  {'err mm':>8}  {'sigma':>7}")

    q_ref   = q_start.copy()
    q_cmd   = q_start.copy()
    prev_qdot  = np.zeros(len(kin.active_joints), dtype=float)
    stable     = 0
    start_t    = time.perf_counter()
    last_print = -1e9

    while True:
        tick_t  = time.perf_counter()
        elapsed = tick_t - start_t

        obs      = robot.get_observation()
        q_actual = kin.clip_joints(_joints_from_obs(obs))

        _, qdot, info = _controller_step(
            kin, q_actual, q_ref, prev_qdot, p_start, target, elapsed, duration
        )
        prev_qdot = qdot

        q_cmd = _write_active_q_rad(
            q_cmd, kin.active_joints,
            _active_q_rad(q_cmd, kin.active_joints) + qdot * dt,
        )
        q_cmd = kin.clip_joints(q_cmd)

        action = {f"{name}.pos": float(q_cmd[index]) for index, name in enumerate(MOTOR_NAMES)}
        action["gripper.pos"]    = float(obs["gripper.pos"])
        action["wrist_roll.pos"] = FIXED_WRIST_ROLL_DEG
        action["wrist_flex.pos"] = kin.fixed_wrist_flex_deg
        robot.send_action(action)

        final_error_m   = float(info["final_error_m"])
        joint_speed_deg = float(np.linalg.norm(np.rad2deg(qdot)))
        if elapsed >= duration and final_error_m < SETTLE_THRESHOLD_M and joint_speed_deg < STOP_JOINT_SPEED_DEG_S:
            stable += 1
            if stable >= STABLE_TICKS:
                print(f"  done    t={elapsed:.3f}s  final={final_error_m * 1000.0:.2f}mm")
                return {"success": True, "final_error_mm": final_error_m * 1000.0, "elapsed_s": elapsed}
        else:
            stable = 0

        if elapsed - last_print >= 0.25:
            print(f"  {elapsed:6.3f}  {final_error_m * 1000.0:8.2f}  {info['sigma_min']:7.4f}")
            last_print = elapsed

        if elapsed > duration + MAX_SETTLE_S:
            print(f"  timeout t={elapsed:.3f}s  final={final_error_m * 1000.0:.2f}mm")
            return {"success": final_error_m < SETTLE_THRESHOLD_M,
                    "final_error_mm": final_error_m * 1000.0, "elapsed_s": elapsed}

        sleep_s = dt - (time.perf_counter() - tick_t)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


# ── REPL ──────────────────────────────────────────────────────────────────────

HELP = """\
  status / s              print current EE position
  go <x|y|z> <meters>    relative move along one axis
  move <x> <y> <z>       absolute move to position (meters)
  quit / q                disconnect and exit"""


def run_repl(robot, kin: SO101Kinematics) -> None:
    print(f"SO-101 EE Controller (fixed wrist)  —  wrist_flex={kin.fixed_wrist_flex_deg:.1f}°  wrist_roll={FIXED_WRIST_ROLL_DEG:.1f}°")
    print("Type 'help' for commands\n")

    while True:
        obs = robot.get_observation()
        q   = _joints_from_obs(obs)
        T   = kin.forward_kinematics(q)
        _print_ee(T)

        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        tokens = raw.lower().split()
        cmd    = tokens[0]

        if cmd in ("q", "quit", "exit"):
            break

        elif cmd in ("s", "status"):
            continue

        elif cmd == "help":
            print(HELP)

        elif cmd == "go":
            if len(tokens) != 3 or tokens[1] not in {"x", "y", "z"}:
                print("  Usage: go <x|y|z> <delta_meters>")
                continue
            try:
                delta = _parse_num(tokens[2])
            except ValueError:
                print("  Delta must be a number")
                continue
            target = T[:3, 3].copy()
            target[{"x": 0, "y": 1, "z": 2}[tokens[1]]] += delta
            smooth_move(robot, kin, target)

        elif cmd == "move":
            nums = []
            for t in tokens[1:]:
                try:
                    nums.append(_parse_num(t))
                except ValueError:
                    pass
            if len(nums) != 3:
                print("  Usage: move <x> <y> <z>  (e.g. move 0.15 0 0.10)")
                continue
            smooth_move(robot, kin, np.array(nums))

        else:
            print(f"  Unknown command '{cmd}'. Type 'help'.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    kin = SO101Kinematics(URDF_PATH)
    print(f"wrist_flex fixed at {kin.fixed_wrist_flex_deg:.2f}°")

    robot = SO101Follower(
        SO101FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
            calibration_dir=CALIBRATION_DIR,
            use_degrees=True,
        )
    )
    robot.connect()

    try:
        run_repl(robot, kin)
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
