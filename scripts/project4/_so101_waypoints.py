from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics


MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINT_NAMES = MOTOR_NAMES[:-1]


@dataclass(frozen=True)
class MotionLimits:
    fps: int = 30
    max_cartesian_step_m: float = 0.012
    max_total_correction_m: float = 0.060
    max_joint_step_deg: float = 8.0
    settle_s: float = 0.35


def joints_from_observation(observation: dict) -> np.ndarray:
    return np.asarray([observation[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float)


class CartesianDeltaMover:
    def __init__(
        self,
        robot,
        urdf_path: str | Path,
        limits: MotionLimits | None = None,
        target_frame_name: str = "gripper_frame_link",
    ) -> None:
        self.robot = robot
        self.limits = limits or MotionLimits()
        self.kinematics = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name=target_frame_name,
            joint_names=ARM_JOINT_NAMES,
        )
        self.total_correction_m = 0.0

    def move_delta(self, dx_m: float, dy_m: float, dz_m: float = 0.0, duration_s: float = 0.6) -> None:
        delta = np.asarray([dx_m, dy_m, dz_m], dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance > self.limits.max_cartesian_step_m:
            raise RuntimeError(
                f"Requested Cartesian step {distance:.4f}m exceeds limit {self.limits.max_cartesian_step_m:.4f}m"
            )
        if self.total_correction_m + distance > self.limits.max_total_correction_m:
            raise RuntimeError("Total visual-servo correction limit exceeded")
        self.total_correction_m += distance

        obs = self.robot.get_observation()
        q = joints_from_observation(obs)
        start_pose = self.kinematics.forward_kinematics(q)
        target_pose = start_pose.copy()
        target_pose[:3, 3] += delta
        steps = max(1, int(duration_s * self.limits.fps))

        previous_q = q.copy()
        for step in range(1, steps + 1):
            loop_start = time.perf_counter()
            alpha = step / steps
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            waypoint_pose = start_pose.copy()
            waypoint_pose[:3, 3] = (1.0 - alpha) * start_pose[:3, 3] + alpha * target_pose[:3, 3]
            target_q = self.kinematics.inverse_kinematics(previous_q, waypoint_pose)
            joint_delta = np.abs(target_q[: len(ARM_JOINT_NAMES)] - previous_q[: len(ARM_JOINT_NAMES)])
            if float(joint_delta.max()) > self.limits.max_joint_step_deg:
                raise RuntimeError(
                    f"IK joint jump {float(joint_delta.max()):.2f}deg exceeds limit "
                    f"{self.limits.max_joint_step_deg:.2f}deg"
                )
            action = {f"{name}.pos": float(target_q[index]) for index, name in enumerate(MOTOR_NAMES)}
            self.robot.send_action(action)
            previous_q = target_q
            sleep_s = 1.0 / self.limits.fps - (time.perf_counter() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
        if self.limits.settle_s > 0:
            time.sleep(self.limits.settle_s)

