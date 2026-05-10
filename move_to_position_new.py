#!/usr/bin/env python
"""
SO-101 move-to-position controller.

This version is designed to keep the useful parts of the two prototypes:

- the smooth task-space path from move_to_position_old.py
- the closed-loop correction from move_to_position.py

but avoids their main failure modes:

- no hard stops at Cartesian waypoints
- no over-constrained "hold all orientation" objective for a position command
- no integral settle loop that can buzz around tiny residual errors
- no dependency on ikpy or placo for the local simulation proof

Core method:

    minimum-jerk Cartesian reference
    + feed-forward Cartesian velocity
    + box-constrained QP differential IK
    + joint-limit/posture bias
    + acceleration-limited joint command smoothing
    + terminal stable hold

Run the simulation comparison:

    python move_to_position_new.py --simulate

Run on hardware:

    python move_to_position_new.py --port COM5 --robot-id my_awesome_follower_arm
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_URDF_PATH = REPO_DIR / "so101_new_calib.urdf"
DEFAULT_OUTPUT_PATH = REPO_DIR / "outputs" / "move_to_position_new_sim.json"
DEFAULT_VIDEO_PATH = REPO_DIR / "outputs" / "move_to_position_new_comparison.mp4"
DEFAULT_MUJOCO_XML = Path(
    r"C:\git_repos\robot-learning\hw2_robot_control_mdps\so101_gym\assets\so100_pos_ctrl.xml"
)

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]
JOINT_INDEX = {name: index for index, name in enumerate(MOTOR_NAMES)}

ROLL_FIXED_DEG = -90.0

WS_MIN = np.array([-0.35, -0.35, -0.10], dtype=float)
WS_MAX = np.array([0.35, 0.35, 0.50], dtype=float)
MAX_MOVE_M = 0.30

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


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


@dataclass
class ControllerConfig:
    fps: int = 60
    solver: str = "qp"
    position_gain: float = 4.0
    max_ee_speed_m_s: float = 0.080
    terminal_ee_speed_m_s: float = 0.018
    min_duration_s: float = 0.45
    max_settle_s: float = 3.0
    settle_threshold_m: float = 0.0020
    stable_ticks: int = 8
    dls_damping: float = 0.025
    singular_threshold: float = 0.035
    singular_damping_gain: float = 0.60
    posture_gain: float = 0.18
    joint_center_gain: float = 0.05
    qp_task_weight: float = 50.0
    qp_damping_weight: float = 0.002
    qp_accel_weight: float = 0.006
    qp_posture_weight: float = 0.012
    max_joint_speed_deg_s: float = 75.0
    max_joint_accel_deg_s2: float = 520.0
    stop_joint_speed_deg_s: float = 1.0
    command_deadband_deg: float = 0.015
    verbose_period_s: float = 0.25


@dataclass
class SimServoConfig:
    fps: int = 60
    max_joint_speed_deg_s: float = 95.0
    deadband_deg: float = 0.030


@dataclass
class SimResult:
    name: str
    success: bool
    final_error_mm: float
    median_error_mm: float
    max_error_mm: float
    path_length_ratio: float
    rms_joint_accel_deg_s2: float
    max_joint_step_deg: float
    min_joint_limit_margin_deg: float
    steps: int


def _parse_vec(text: str | None, default: Iterable[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.asarray(tuple(default), dtype=float)
    return np.asarray([float(x) for x in text.split()], dtype=float)


def _rx(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _ry(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rz(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = _rpy_to_matrix(rpy)
    out[:3, 3] = xyz
    return out


def _rot_error(R_target: np.ndarray, R_curr: np.ndarray) -> np.ndarray:
    R_err = R_target @ R_curr.T
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(trace))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    return (angle / (2.0 * math.sin(angle))) * np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ],
        dtype=float,
    )


def _clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > max_norm > 0.0:
        return vec * (max_norm / norm)
    return vec


def _minimum_jerk(tau: float) -> tuple[float, float]:
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds_dtau = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    return s, ds_dtau


def _parse_num(token: str) -> float:
    match = _NUM_RE.search(token)
    if match is None:
        raise ValueError(token)
    return float(match.group())


def _joints_from_obs(obs: dict) -> np.ndarray:
    return np.asarray([obs[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float)


def _print_ee(T: np.ndarray, prefix: str = "") -> None:
    p = T[:3, 3]
    print(f"{prefix}EE -> x={p[0]:+.4f}m  y={p[1]:+.4f}m  z={p[2]:+.4f}m")


def _clamp_target(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    target = np.clip(target.astype(float), WS_MIN, WS_MAX)
    delta = target - current
    dist = float(np.linalg.norm(delta))
    if dist > MAX_MOVE_M:
        target = current + delta * (MAX_MOVE_M / dist)
        print(f"  [safety] clamped move to {MAX_MOVE_M:.2f}m")
    return target


class SO101Kinematics:
    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF_PATH,
        base_link: str = "base_link",
        target_link: str = "gripper_frame_link",
    ) -> None:
        self.urdf_path = Path(urdf_path)
        self.base_link = base_link
        self.target_link = target_link
        self.joints = self._load_joints(self.urdf_path)
        self.chain = self._chain_to_target()
        self.active_joints = [name for name in ARM_JOINTS if name in self.joints]

    @staticmethod
    def _load_joints(path: Path) -> dict[str, JointSpec]:
        root = ET.parse(path).getroot()
        joints: dict[str, JointSpec] = {}
        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            joint_type = joint.attrib.get("type", "fixed")
            origin = joint.find("origin")
            parent = joint.find("parent")
            child = joint.find("child")
            axis = joint.find("axis")
            limit = joint.find("limit")
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

    def _q_map(self, joint_pos_deg: np.ndarray | dict[str, float]) -> dict[str, float]:
        if isinstance(joint_pos_deg, dict):
            return {name: math.radians(float(joint_pos_deg.get(name, 0.0))) for name in ARM_JOINTS}
        q = np.asarray(joint_pos_deg, dtype=float)
        return {name: math.radians(float(q[index])) for name, index in JOINT_INDEX.items() if name in ARM_JOINTS}

    def forward_kinematics(self, joint_pos_deg: np.ndarray | dict[str, float]) -> np.ndarray:
        q_by_name = self._q_map(joint_pos_deg)
        T = np.eye(4, dtype=float)
        for joint in self.chain:
            T = T @ _transform(joint.xyz, joint.rpy)
            if joint.joint_type in {"revolute", "continuous"}:
                R = _axis_angle_to_matrix(joint.axis, q_by_name.get(joint.name, 0.0))
                T_joint = np.eye(4, dtype=float)
                T_joint[:3, :3] = R
                T = T @ T_joint
        return T

    def bounds_deg(self, joint_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        names = joint_names or self.active_joints
        lower = []
        upper = []
        for name in names:
            spec = self.joints[name]
            lo = -np.inf if spec.lower is None else math.degrees(spec.lower)
            hi = np.inf if spec.upper is None else math.degrees(spec.upper)
            lower.append(lo)
            upper.append(hi)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def clip_joints(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        q = np.asarray(joint_pos_deg, dtype=float).copy()
        for name in ARM_JOINTS:
            spec = self.joints.get(name)
            if spec is None:
                continue
            index = JOINT_INDEX[name]
            if spec.lower is not None:
                q[index] = max(q[index], math.degrees(spec.lower))
            if spec.upper is not None:
                q[index] = min(q[index], math.degrees(spec.upper))
        return q

    def position_jacobian(
        self,
        joint_pos_deg: np.ndarray,
        active_joints: list[str] | None = None,
        eps_rad: float = 1e-4,
    ) -> np.ndarray:
        names = active_joints or self.active_joints
        q = np.asarray(joint_pos_deg, dtype=float)
        J = np.zeros((3, len(names)), dtype=float)
        eps_deg = math.degrees(eps_rad)
        for col, name in enumerate(names):
            q_plus = q.copy()
            q_minus = q.copy()
            idx = JOINT_INDEX[name]
            q_plus[idx] += eps_deg
            q_minus[idx] -= eps_deg
            p_plus = self.forward_kinematics(q_plus)[:3, 3]
            p_minus = self.forward_kinematics(q_minus)[:3, 3]
            J[:, col] = (p_plus - p_minus) / (2.0 * eps_rad)
        return J


def _planned_duration(distance_m: float, cfg: ControllerConfig) -> float:
    if distance_m <= 1e-9:
        return cfg.min_duration_s
    return max(cfg.min_duration_s, 1.875 * distance_m / cfg.max_ee_speed_m_s)


def _active_q_rad(q_deg: np.ndarray, active_joints: list[str]) -> np.ndarray:
    return np.asarray([math.radians(float(q_deg[JOINT_INDEX[name]])) for name in active_joints], dtype=float)


def _write_active_q_rad(q_deg: np.ndarray, active_joints: list[str], q_active_rad: np.ndarray) -> np.ndarray:
    q = np.asarray(q_deg, dtype=float).copy()
    for value, name in zip(q_active_rad, active_joints):
        q[JOINT_INDEX[name]] = math.degrees(float(value))
    return q


def _joint_limit_terms(
    kin: SO101Kinematics,
    q_deg: np.ndarray,
    active_joints: list[str],
    q_ref_deg: np.ndarray,
    cfg: ControllerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_rad = _active_q_rad(q_deg, active_joints)
    q_ref_rad = _active_q_rad(q_ref_deg, active_joints)
    lo_deg, hi_deg = kin.bounds_deg(active_joints)
    lo_rad = np.deg2rad(lo_deg)
    hi_rad = np.deg2rad(hi_deg)
    center_rad = 0.5 * (lo_rad + hi_rad)
    half_range_rad = np.maximum(0.5 * (hi_rad - lo_rad), np.deg2rad(5.0))
    normalized = np.clip((q_rad - center_rad) / half_range_rad, -1.5, 1.5)
    qdot_posture = -cfg.posture_gain * (q_rad - q_ref_rad)
    qdot_center = -cfg.joint_center_gain * normalized**3
    return q_rad, lo_rad, hi_rad, qdot_posture + qdot_center


def _solve_box_qp(H: np.ndarray, g: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, int]:
    """Solve a small convex box-QP with an active-set clamp loop.

    Objective: 0.5 x.T H x + g.T x, subject to lower <= x <= upper.
    This is intentionally tiny and dependency-free for the 5-DoF SO-101 case.
    """
    n = int(g.size)
    H = H + 1e-9 * np.eye(n)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    fixed = np.zeros(n, dtype=bool)
    x = np.clip(np.zeros(n, dtype=float), lower, upper)

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

        below = x < lower
        above = x > upper
        violation = np.maximum(lower - x, x - upper)
        if not np.any(below | above):
            return x, iteration + 1

        idx = int(np.argmax(violation))
        x[idx] = lower[idx] if below[idx] else upper[idx]
        fixed[idx] = True

    return np.clip(x, lower, upper), n + 1


def _joint_velocity_bounds(
    q_rad: np.ndarray,
    lo_rad: np.ndarray,
    hi_rad: np.ndarray,
    prev_qdot_rad_s: np.ndarray,
    cfg: ControllerConfig,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    speed = math.radians(cfg.max_joint_speed_deg_s)
    accel_delta = math.radians(cfg.max_joint_accel_deg_s2)
    lower = np.maximum(-speed, (lo_rad - q_rad) / dt)
    upper = np.minimum(speed, (hi_rad - q_rad) / dt)
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
    cfg: ControllerConfig,
    dt: float,
) -> tuple[np.ndarray, dict[str, float]]:
    J = kin.position_jacobian(q_deg, active_joints)
    singular_values = np.linalg.svd(J, compute_uv=False)
    sigma_min = float(singular_values[-1]) if len(singular_values) else 0.0
    singular_boost = max(0.0, cfg.singular_threshold - sigma_min)
    damping_weight = cfg.qp_damping_weight + cfg.singular_damping_gain * singular_boost * singular_boost

    q_rad, lo_rad, hi_rad, qdot_pref = _joint_limit_terms(kin, q_deg, active_joints, q_ref_deg, cfg)
    lower, upper = _joint_velocity_bounds(q_rad, lo_rad, hi_rad, prev_qdot_rad_s, cfg, dt)

    n = len(active_joints)
    H = (
        cfg.qp_task_weight * (J.T @ J)
        + damping_weight * np.eye(n)
        + cfg.qp_accel_weight * np.eye(n)
        + cfg.qp_posture_weight * np.eye(n)
    )
    g = (
        -cfg.qp_task_weight * (J.T @ v_task)
        - cfg.qp_accel_weight * prev_qdot_rad_s
        - cfg.qp_posture_weight * qdot_pref
    )
    qdot, qp_iters = _solve_box_qp(H, g, lower, upper)
    return qdot, {
        "solver": "qp",
        "sigma_min": sigma_min,
        "damping": damping_weight,
        "qp_iters": float(qp_iters),
    }


def _bounded_dls_qdot(
    kin: SO101Kinematics,
    q_deg: np.ndarray,
    v_task: np.ndarray,
    active_joints: list[str],
    q_ref_deg: np.ndarray,
    prev_qdot_rad_s: np.ndarray,
    cfg: ControllerConfig,
    dt: float,
) -> tuple[np.ndarray, dict[str, float]]:
    J = kin.position_jacobian(q_deg, active_joints)
    singular_values = np.linalg.svd(J, compute_uv=False)
    sigma_min = float(singular_values[-1]) if len(singular_values) else 0.0
    damping = cfg.dls_damping + max(0.0, cfg.singular_threshold - sigma_min) * cfg.singular_damping_gain

    A = J @ J.T + (damping * damping) * np.eye(3)
    J_pinv = J.T @ np.linalg.solve(A, np.eye(3))
    qdot_task = J_pinv @ v_task

    q_rad, lo_rad, hi_rad, qdot_pref = _joint_limit_terms(kin, q_deg, active_joints, q_ref_deg, cfg)

    nullspace = np.eye(len(active_joints)) - J_pinv @ J
    qdot = qdot_task + nullspace @ qdot_pref
    lower, upper = _joint_velocity_bounds(q_rad, lo_rad, hi_rad, prev_qdot_rad_s, cfg, dt)
    qdot = np.clip(qdot, lower, upper)

    return qdot, {"solver": "dls", "sigma_min": sigma_min, "damping": damping, "qp_iters": 0.0}


def _solve_qdot(
    kin: SO101Kinematics,
    q_deg: np.ndarray,
    v_task: np.ndarray,
    active_joints: list[str],
    q_ref_deg: np.ndarray,
    prev_qdot_rad_s: np.ndarray,
    cfg: ControllerConfig,
    dt: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if cfg.solver == "dls":
        return _bounded_dls_qdot(kin, q_deg, v_task, active_joints, q_ref_deg, prev_qdot_rad_s, cfg, dt)
    if cfg.solver != "qp":
        raise ValueError(f"Unsupported solver {cfg.solver!r}; expected 'qp' or 'dls'")
    return _qp_qdot(kin, q_deg, v_task, active_joints, q_ref_deg, prev_qdot_rad_s, cfg, dt)


def _controller_step(
    kin: SO101Kinematics,
    q_deg: np.ndarray,
    q_ref_deg: np.ndarray,
    prev_qdot_rad_s: np.ndarray,
    p_start: np.ndarray,
    target_pos: np.ndarray,
    elapsed_s: float,
    duration_s: float,
    cfg: ControllerConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    dt = 1.0 / cfg.fps
    T_curr = kin.forward_kinematics(q_deg)
    p_curr = T_curr[:3, 3]
    tau = elapsed_s / max(duration_s, 1e-6)
    s, ds_dtau = _minimum_jerk(tau)
    final_phase = tau >= 1.0

    p_ref = p_start + s * (target_pos - p_start)
    if final_phase:
        p_ref = target_pos
        v_ff = np.zeros(3)
    else:
        v_ff = (target_pos - p_start) * (ds_dtau / duration_s)

    error = p_ref - p_curr
    final_error = target_pos - p_curr
    speed_cap = cfg.terminal_ee_speed_m_s if final_phase or np.linalg.norm(final_error) < 0.015 else cfg.max_ee_speed_m_s
    v_task = _clamp_norm(v_ff + cfg.position_gain * error, speed_cap)

    qdot, info = _solve_qdot(
        kin=kin,
        q_deg=q_deg,
        v_task=v_task,
        active_joints=kin.active_joints,
        q_ref_deg=q_ref_deg,
        prev_qdot_rad_s=prev_qdot_rad_s,
        cfg=cfg,
        dt=dt,
    )
    q_next = _write_active_q_rad(q_deg, kin.active_joints, _active_q_rad(q_deg, kin.active_joints) + qdot * dt)
    q_next = kin.clip_joints(q_next)
    info.update(
        {
            "tracking_error_m": float(np.linalg.norm(error)),
            "final_error_m": float(np.linalg.norm(final_error)),
            "speed_cap_m_s": speed_cap,
            "tau": min(tau, 1.0),
        }
    )
    return q_next, qdot, info


def move_to_position(
    robot,
    kin: SO101Kinematics,
    target_pos: np.ndarray,
    cfg: ControllerConfig | None = None,
    duration_s: float | None = None,
    verbose: bool = True,
) -> dict[str, float | str | bool]:
    cfg = cfg or ControllerConfig()
    dt = 1.0 / cfg.fps
    obs = robot.get_observation()
    q_start = kin.clip_joints(_joints_from_obs(obs))
    T_start = kin.forward_kinematics(q_start)
    p_start = T_start[:3, 3].copy()
    target_pos = _clamp_target(np.asarray(target_pos, dtype=float), p_start)
    distance = float(np.linalg.norm(target_pos - p_start))
    duration = float(duration_s if duration_s is not None else _planned_duration(distance, cfg))

    if distance < 1e-4:
        return {"success": True, "status": "already_at_target", "final_error_mm": 0.0, "elapsed_s": 0.0}

    if verbose:
        print(f"  Moving {distance * 100.0:.1f} cm in {duration:.2f}s reference time")
        print(f"  {'t(s)':>6}  {'err mm':>8}  {'sigma':>7}  {'solver':>6}")

    q_ref = q_start.copy()
    prev_qdot = np.zeros(len(kin.active_joints), dtype=float)
    last_sent = q_start.copy()
    stable = 0
    start_t = time.perf_counter()
    last_print = -1e9

    while True:
        tick_t = time.perf_counter()
        elapsed = tick_t - start_t
        obs = robot.get_observation()
        q_actual = kin.clip_joints(_joints_from_obs(obs))

        q_cmd, qdot, info = _controller_step(
            kin=kin,
            q_deg=q_actual,
            q_ref_deg=q_ref,
            prev_qdot_rad_s=prev_qdot,
            p_start=p_start,
            target_pos=target_pos,
            elapsed_s=elapsed,
            duration_s=duration,
            cfg=cfg,
        )
        prev_qdot = qdot

        max_command_delta = float(np.max(np.abs(q_cmd[:5] - last_sent[:5])))
        if max_command_delta >= cfg.command_deadband_deg or info["final_error_m"] > cfg.settle_threshold_m:
            action = {f"{name}.pos": float(q_cmd[index]) for index, name in enumerate(MOTOR_NAMES)}
            action["gripper.pos"] = float(obs["gripper.pos"])
            robot.send_action(action)
            last_sent = q_cmd

        final_error_m = float(info["final_error_m"])
        joint_speed_deg_s = float(np.linalg.norm(np.rad2deg(qdot)))
        if elapsed >= duration and final_error_m < cfg.settle_threshold_m and joint_speed_deg_s < cfg.stop_joint_speed_deg_s:
            stable += 1
            if stable >= cfg.stable_ticks:
                if verbose:
                    print(f"  done    t={elapsed:.3f}s  final={final_error_m * 1000.0:.2f}mm")
                return {
                    "success": True,
                    "status": "done",
                    "final_error_mm": final_error_m * 1000.0,
                    "elapsed_s": elapsed,
                }
        else:
            stable = 0

        if verbose and elapsed - last_print >= cfg.verbose_period_s:
            print(f"  {elapsed:6.3f}  {final_error_m * 1000.0:8.2f}  {info['sigma_min']:7.4f}  {info['solver']:>6}")
            last_print = elapsed

        if elapsed > duration + cfg.max_settle_s:
            if verbose:
                print(f"  timeout t={elapsed:.3f}s  final={final_error_m * 1000.0:.2f}mm")
            return {
                "success": final_error_m < cfg.settle_threshold_m,
                "status": "timeout",
                "final_error_mm": final_error_m * 1000.0,
                "elapsed_s": elapsed,
            }

        sleep_s = dt - (time.perf_counter() - tick_t)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def _servo_step(q_actual: np.ndarray, q_cmd: np.ndarray, dt: float, servo: SimServoConfig) -> np.ndarray:
    q = q_actual.copy()
    max_step = servo.max_joint_speed_deg_s * dt
    for index, name in enumerate(ARM_JOINTS):
        delta = float(q_cmd[index] - q[index])
        if abs(delta) < servo.deadband_deg:
            continue
        q[index] += float(np.clip(delta, -max_step, max_step))
    q[5] = q_cmd[5]
    return q


def _trace_metrics(
    name: str,
    kin: SO101Kinematics,
    trace_q: list[np.ndarray],
    target_pos: np.ndarray,
    dt: float,
    threshold_m: float = 0.003,
) -> SimResult:
    q_arr = np.asarray(trace_q, dtype=float)
    positions = np.asarray([kin.forward_kinematics(q)[:3, 3] for q in q_arr], dtype=float)
    errors = np.linalg.norm(target_pos - positions, axis=1)
    direct = float(np.linalg.norm(target_pos - positions[0]))
    path_len = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(positions) > 1 else 0.0
    joint_steps = np.diff(q_arr[:, :5], axis=0) if len(q_arr) > 1 else np.zeros((0, 5))
    joint_speed = joint_steps / dt if len(joint_steps) else np.zeros((0, 5))
    joint_accel = np.diff(joint_speed, axis=0) / dt if len(joint_speed) > 1 else np.zeros((0, 5))
    lo, hi = kin.bounds_deg(ARM_JOINTS)
    margins = np.minimum(q_arr[:, :5] - lo, hi - q_arr[:, :5])
    return SimResult(
        name=name,
        success=bool(errors[-1] < threshold_m),
        final_error_mm=float(errors[-1] * 1000.0),
        median_error_mm=float(np.median(errors) * 1000.0),
        max_error_mm=float(np.max(errors) * 1000.0),
        path_length_ratio=float(path_len / direct) if direct > 1e-9 else 1.0,
        rms_joint_accel_deg_s2=float(np.sqrt(np.mean(joint_accel**2))) if joint_accel.size else 0.0,
        max_joint_step_deg=float(np.max(np.abs(joint_steps))) if joint_steps.size else 0.0,
        min_joint_limit_margin_deg=float(np.min(margins)),
        steps=len(trace_q),
    )


def _solve_position_ik(
    kin: SO101Kinematics,
    q_seed: np.ndarray,
    target_pos: np.ndarray,
    active_joints: list[str],
    fixed: dict[str, float] | None = None,
    iterations: int = 40,
    damping: float = 0.025,
    max_step_deg: float = 6.0,
) -> np.ndarray:
    q = kin.clip_joints(q_seed)
    fixed = fixed or {}
    for name, value in fixed.items():
        q[JOINT_INDEX[name]] = value
    for _ in range(iterations):
        p = kin.forward_kinematics(q)[:3, 3]
        error = target_pos - p
        if float(np.linalg.norm(error)) < 5e-4:
            break
        J = kin.position_jacobian(q, active_joints)
        A = J @ J.T + (damping * damping) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, error)
        dq_deg = np.clip(np.rad2deg(dq), -max_step_deg, max_step_deg)
        for value, name in zip(dq_deg, active_joints):
            q[JOINT_INDEX[name]] += float(value)
        for name, value in fixed.items():
            q[JOINT_INDEX[name]] = value
        q = kin.clip_joints(q)
    return q


def _simulate_new(
    kin: SO101Kinematics,
    q0: np.ndarray,
    target_pos: np.ndarray,
    cfg: ControllerConfig,
    servo: SimServoConfig,
) -> list[np.ndarray]:
    dt = 1.0 / cfg.fps
    q = kin.clip_joints(q0)
    p_start = kin.forward_kinematics(q)[:3, 3].copy()
    target = _clamp_target(target_pos, p_start)
    duration = _planned_duration(float(np.linalg.norm(target - p_start)), cfg)
    prev_qdot = np.zeros(len(kin.active_joints), dtype=float)
    q_ref = q.copy()
    trace = [q.copy()]
    stable = 0
    max_steps = int((duration + cfg.max_settle_s) * cfg.fps)
    for step in range(1, max_steps + 1):
        q_cmd, qdot, info = _controller_step(kin, q, q_ref, prev_qdot, p_start, target, step * dt, duration, cfg)
        prev_qdot = qdot
        q = _servo_step(q, q_cmd, dt, servo)
        q = kin.clip_joints(q)
        trace.append(q.copy())
        final_error = float(info["final_error_m"])
        speed = float(np.linalg.norm(np.rad2deg(qdot)))
        if step * dt >= duration and final_error < cfg.settle_threshold_m and speed < cfg.stop_joint_speed_deg_s:
            stable += 1
            if stable >= cfg.stable_ticks:
                break
        else:
            stable = 0
    return trace


def _simulate_current_jacobian(
    kin: SO101Kinematics,
    q0: np.ndarray,
    target_pos: np.ndarray,
    servo: SimServoConfig,
) -> list[np.ndarray]:
    dt = 1.0 / 30.0
    q = kin.clip_joints(q0)
    R_target = kin.forward_kinematics(q)[:3, :3].copy()
    active = ARM_JOINTS
    trace = [q.copy()]
    for _ in range(300):
        T = kin.forward_kinematics(q)
        pos_err = target_pos - T[:3, 3]
        rot_err = _rot_error(R_target, T[:3, :3])
        if float(np.linalg.norm(pos_err)) < 0.003 and float(np.linalg.norm(rot_err[:2])) < 0.05:
            break

        J = np.zeros((5, 5), dtype=float)
        eps_rad = 1e-4
        eps_deg = math.degrees(eps_rad)
        for col, name in enumerate(active):
            q_plus = q.copy()
            q_plus[JOINT_INDEX[name]] += eps_deg
            T_plus = kin.forward_kinematics(q_plus)
            J[:3, col] = (T_plus[:3, 3] - T[:3, 3]) / eps_rad
            R_diff = T_plus[:3, :3] @ T[:3, :3].T
            J[3:, col] = (_rot_error(R_diff, np.eye(3)) / eps_rad)[:2]

        err = np.concatenate([pos_err, 0.3 * rot_err[:2]])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(5), err)
        q_cmd = q.copy()
        q_cmd[:5] += np.rad2deg(0.8 * dq)
        q_cmd = kin.clip_joints(q_cmd)
        q = _servo_step(q, q_cmd, dt, servo)
        trace.append(q.copy())
    return trace


def _simulate_old_waypoint_pid(
    kin: SO101Kinematics,
    q0: np.ndarray,
    target_pos: np.ndarray,
    servo: SimServoConfig,
) -> list[np.ndarray]:
    dt = 1.0 / 30.0
    q = kin.clip_joints(q0)
    q[JOINT_INDEX["wrist_roll"]] = ROLL_FIXED_DEG
    p_start = kin.forward_kinematics(q)[:3, 3].copy()
    target = _clamp_target(target_pos, p_start)
    trace = [q.copy()]
    active4 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
    q_seed = q.copy()
    for step in range(1, int(2.0 / dt) + 1):
        alpha = step / int(2.0 / dt)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        p_wp = (1.0 - alpha) * p_start + alpha * target
        q_cmd = _solve_position_ik(
            kin,
            q_seed,
            p_wp,
            active_joints=active4,
            fixed={"wrist_roll": ROLL_FIXED_DEG},
            iterations=18,
        )
        q_seed = q_cmd.copy()
        q = _servo_step(q, q_cmd, dt, servo)
        trace.append(q.copy())

    integral = np.zeros(3)
    prev_error = np.zeros(3)
    for _ in range(60):
        p = kin.forward_kinematics(q)[:3, 3]
        error = target - p
        if float(np.linalg.norm(error)) < 0.003:
            break
        integral += error * dt
        correction = error + 5.0 * integral + 0.0 * ((error - prev_error) / dt)
        prev_error = error
        correction = _clamp_norm(correction, 0.05)
        q_cmd = _solve_position_ik(
            kin,
            q,
            p + correction,
            active_joints=active4,
            fixed={"wrist_roll": ROLL_FIXED_DEG},
            iterations=18,
        )
        q = _servo_step(q, q_cmd, dt, servo)
        trace.append(q.copy())
    return trace


def _sample_cases(kin: SO101Kinematics, count: int, seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    lo, hi = kin.bounds_deg(ARM_JOINTS)
    q_start = np.array([0.0, -68.0, 84.0, 34.0, ROLL_FIXED_DEG, 80.0], dtype=float)
    start_pos = kin.forward_kinematics(q_start)[:3, 3]
    cases: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("small_x_5mm", q_start, start_pos + np.array([0.005, 0.0, 0.0])),
        ("small_xyz_8mm", q_start, start_pos + np.array([0.004, -0.003, 0.006])),
        ("vertical_25mm", q_start, start_pos + np.array([0.0, 0.0, 0.025])),
    ]
    attempts = 0
    while len(cases) < count and attempts < count * 40:
        attempts += 1
        q_goal = q_start.copy()
        q_goal[:5] = rng.uniform(lo + 0.18 * (hi - lo), hi - 0.18 * (hi - lo))
        p_goal = kin.forward_kinematics(q_goal)[:3, 3]
        dist = float(np.linalg.norm(p_goal - start_pos))
        if 0.015 <= dist <= 0.22 and np.all(p_goal >= WS_MIN) and np.all(p_goal <= WS_MAX):
            cases.append((f"random_{len(cases) - 2:02d}", q_start.copy(), p_goal))
    return cases


def _q_deg_to_mujoco_qpos(model, q_deg: np.ndarray) -> np.ndarray:
    qpos = np.deg2rad(np.asarray(q_deg[: model.nq], dtype=float))
    if model.njnt == model.nq:
        qpos = np.clip(qpos, model.jnt_range[:, 0], model.jnt_range[:, 1])
    return qpos


def _trace_at_time(trace: list[np.ndarray], dt: float, t_s: float) -> np.ndarray:
    if not trace:
        raise ValueError("Cannot sample an empty trace")
    index = min(int(round(t_s / dt)), len(trace) - 1)
    return trace[index]


def _render_mujoco_comparison_video(
    mujoco_xml: str | Path,
    traces: dict[str, list[np.ndarray]],
    trace_dt: dict[str, float],
    target_pos: np.ndarray,
    output_path: str | Path,
    fps: int = 30,
    width: int = 360,
    height: int = 300,
) -> Path | None:
    xml_path = Path(mujoco_xml)
    if not xml_path.exists():
        print(f"  [video] MuJoCo XML not found: {xml_path}")
        return None

    try:
        import cv2
        import mujoco
    except Exception as exc:
        print(f"  [video] skipping MuJoCo video; missing dependency: {exc}")
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    if target_body_id >= 0:
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) == target_body_id:
                model.geom_size[geom_id, 0] = 0.028
                model.geom_rgba[geom_id] = np.array([0.0, 1.0, 0.15, 1.0], dtype=np.float32)

    renderer = mujoco.Renderer(model, width=width, height=height)
    names = [name for name in ("old", "current", "new") if name in traces]
    total_s = max((len(traces[name]) - 1) * trace_dt[name] for name in names)
    frame_count = max(1, int(math.ceil(total_s * fps)))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width * len(names), height),
    )
    if not writer.isOpened():
        print(f"  [video] could not open writer for {output}")
        return None

    rgba = {
        "old": (230, 140, 20),
        "current": (30, 130, 230),
        "new": (30, 170, 80),
    }
    try:
        for frame_index in range(frame_count):
            t_s = frame_index / fps
            panels = []
            for name in names:
                q_deg = _trace_at_time(traces[name], trace_dt[name], t_s)
                data.qpos[:] = _q_deg_to_mujoco_qpos(model, q_deg)
                data.qvel[:] = 0.0
                if model.nmocap > 0:
                    data.mocap_pos[0] = np.asarray(target_pos, dtype=float)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                color = rgba.get(name, (255, 255, 255))
                cv2.putText(
                    bgr,
                    name,
                    (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr,
                    "GOAL = bright green sphere",
                    (14, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (90, 255, 90),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr,
                    f"t={t_s:4.2f}s",
                    (14, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                panels.append(bgr)
            writer.write(np.hstack(panels))
    finally:
        writer.release()
        renderer.close()

    print(f"  [video] wrote {output}")
    return output


def run_simulation(args: argparse.Namespace) -> dict:
    kin = SO101Kinematics(args.urdf_path)
    cfg = ControllerConfig(fps=args.sim_fps)
    servo = SimServoConfig(fps=args.sim_fps)
    cases = _sample_cases(kin, args.sim_cases, args.seed)
    rows = []
    by_controller: dict[str, list[SimResult]] = {"old": [], "current": [], "new": []}
    video_case = None

    for case_name, q0, target in cases:
        old_trace = _simulate_old_waypoint_pid(kin, q0.copy(), target, servo)
        current_trace = _simulate_current_jacobian(kin, q0.copy(), target, servo)
        new_trace = _simulate_new(kin, q0.copy(), target, cfg, servo)
        if case_name == args.video_case or video_case is None:
            video_case = {
                "case": case_name,
                "target": target,
                "traces": {"old": old_trace, "current": current_trace, "new": new_trace},
                "trace_dt": {"old": 1.0 / 30.0, "current": 1.0 / 30.0, "new": 1.0 / cfg.fps},
            }
        metrics = [
            _trace_metrics("old", kin, old_trace, target, 1.0 / 30.0),
            _trace_metrics("current", kin, current_trace, target, 1.0 / 30.0),
            _trace_metrics("new", kin, new_trace, target, 1.0 / cfg.fps),
        ]
        for result in metrics:
            by_controller[result.name].append(result)
            rows.append({"case": case_name, **asdict(result)})

    summary = {}
    for name, results in by_controller.items():
        summary[name] = {
            "success_rate": float(np.mean([r.success for r in results])),
            "mean_final_error_mm": float(np.mean([r.final_error_mm for r in results])),
            "median_final_error_mm": float(np.median([r.final_error_mm for r in results])),
            "mean_rms_joint_accel_deg_s2": float(np.mean([r.rms_joint_accel_deg_s2 for r in results])),
            "mean_path_length_ratio": float(np.mean([r.path_length_ratio for r in results])),
            "mean_max_joint_step_deg": float(np.mean([r.max_joint_step_deg for r in results])),
        }

    report = {
        "urdf_path": str(Path(args.urdf_path).resolve()),
        "simulation": {
            "cases": len(cases),
            "seed": args.seed,
            "servo": asdict(servo),
            "controller": asdict(cfg),
            "video_case": None if video_case is None else video_case["case"],
        },
        "summary": summary,
        "rows": rows,
    }

    if not args.no_video and video_case is not None:
        video_path = _render_mujoco_comparison_video(
            mujoco_xml=args.mujoco_xml,
            traces=video_case["traces"],
            trace_dt=video_case["trace_dt"],
            target_pos=video_case["target"],
            output_path=args.video_output,
            fps=args.video_fps,
        )
        if video_path is not None:
            report["simulation"]["video_path"] = str(video_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSimulation summary")
    print(f"  cases: {len(cases)}")
    print(f"  output: {output_path}")
    print()
    print(f"{'controller':>10}  {'success':>8}  {'final mm':>9}  {'accel rms':>10}  {'path':>6}  {'max step':>8}")
    for name in ("old", "current", "new"):
        item = summary[name]
        print(
            f"{name:>10}  "
            f"{item['success_rate'] * 100.0:7.1f}%  "
            f"{item['mean_final_error_mm']:9.2f}  "
            f"{item['mean_rms_joint_accel_deg_s2']:10.1f}  "
            f"{item['mean_path_length_ratio']:6.2f}  "
            f"{item['mean_max_joint_step_deg']:8.2f}"
        )
    return report


HELP = """\
  status / s              print current EE position
  go <x|y|z> <meters>    relative move along one axis
  move <x> <y> <z>       absolute move to position (meters)
  quit / q                disconnect and exit"""


def run_repl(robot, kin: SO101Kinematics, cfg: ControllerConfig) -> None:
    print("SO-101 EE Controller (minimum-jerk + bounded DLS) -- type 'help' for commands\n")
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
        if cmd in {"q", "quit", "exit"}:
            break
        if cmd in {"s", "status"}:
            continue
        if cmd == "help":
            print(HELP)
            continue
        if cmd == "go":
            if len(tokens) != 3 or tokens[1] not in {"x", "y", "z"}:
                print("  Usage: go <x|y|z> <delta_meters>")
                continue
            axis = {"x": 0, "y": 1, "z": 2}[tokens[1]]
            try:
                delta = _parse_num(tokens[2])
            except ValueError:
                print("  Delta must be a number")
                continue
            target = T[:3, 3].copy()
            target[axis] += delta
            move_to_position(robot, kin, target, cfg)
            continue
        if cmd == "move":
            nums = []
            for token in tokens[1:]:
                try:
                    nums.append(_parse_num(token))
                except ValueError:
                    pass
            if len(nums) != 3:
                print("  Usage: move <x> <y> <z>")
                continue
            move_to_position(robot, kin, np.asarray(nums, dtype=float), cfg)
            continue
        print(f"  Unknown command '{cmd}'. Type 'help'.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SO-101 move-to-position controller and simulation benchmark.")
    parser.add_argument("--simulate", action="store_true", help="Run the offline simulation comparison and exit.")
    parser.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH), help="Path to so101_new_calib.urdf.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Simulation JSON output path.")
    parser.add_argument("--sim-cases", type=int, default=18, help="Number of simulation cases.")
    parser.add_argument("--sim-fps", type=int, default=60, help="Simulation/control FPS for the new controller.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for simulation target generation.")
    parser.add_argument("--no-video", action="store_true", help="Skip MuJoCo comparison video generation.")
    parser.add_argument("--video-output", default=str(DEFAULT_VIDEO_PATH), help="MuJoCo comparison MP4 output path.")
    parser.add_argument("--video-fps", type=int, default=30, help="Output video frame rate.")
    parser.add_argument("--video-case", default="vertical_25mm", help="Simulation case name to render.")
    parser.add_argument("--mujoco-xml", default=str(DEFAULT_MUJOCO_XML), help="MuJoCo XML used for video playback.")
    parser.add_argument("--port", default="COM5", help="SO-101 follower serial port.")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm", help="LeRobot robot id.")
    parser.add_argument("--calibration-dir", default=str(Path.home() / "robots"), help="LeRobot calibration directory.")
    parser.add_argument("--fps", type=int, default=60, help="Hardware control FPS.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.simulate:
        run_simulation(args)
        return

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    kin = SO101Kinematics(args.urdf_path)
    cfg = ControllerConfig(fps=args.fps)
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            calibration_dir=Path(args.calibration_dir),
            use_degrees=True,
        )
    )
    robot.connect()
    try:
        run_repl(robot, kin, cfg)
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
