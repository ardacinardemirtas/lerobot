#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _keyboard_targets import TargetMapError, load_target_map, normalize_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameEstimate:
    target_px: np.ndarray
    tip_px: np.ndarray
    error_px: np.ndarray
    fit_reason: str
    tip_reason: str
    frame_path: Path


def add_atakan_code_to_path(path: str | Path) -> None:
    atakan = Path(path).resolve()
    if str(atakan) not in sys.path:
        sys.path.insert(0, str(atakan))


def save_frame(frame, path: Path) -> None:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 frame, got shape {arr.shape}")
    image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


class LiveFrameEstimator:
    def __init__(self, atakan_code_dir: str | Path, env_file: str | Path, threshold: float) -> None:
        add_atakan_code_to_path(atakan_code_dir)
        from keyboard_hover.detector import DetectorClient
        from keyboard_hover.grid_fit import fit_key_grid
        from keyboard_hover.tip import detect_tip

        self.client = DetectorClient(env_file=Path(env_file))
        self.fit_key_grid = fit_key_grid
        self.detect_tip = detect_tip
        self.threshold = threshold

    def estimate(self, frame, frame_path: Path, target_key: str) -> FrameEstimate:
        save_frame(frame, frame_path)
        detection = self.client.detect_image(frame_path, confidence=self.threshold)
        fit = self.fit_key_grid(detection.predictions)
        if not fit.accepted:
            raise RuntimeError(f"Key-grid fit rejected: {fit.reason}")
        label = normalize_key(target_key)
        if label not in fit.key_targets:
            raise RuntimeError(f"Target {target_key!r} unavailable in live frame")
        tip = self.detect_tip(frame_path)
        if not tip.accepted or tip.center_px is None:
            raise RuntimeError(f"Tip detection rejected: {tip.reason}")
        target_px = np.asarray(fit.key_targets[label].center_px, dtype=float)
        tip_px = np.asarray(tip.center_px, dtype=float)
        return FrameEstimate(
            target_px=target_px,
            tip_px=tip_px,
            error_px=target_px - tip_px,
            fit_reason=fit.reason,
            tip_reason=tip.reason,
            frame_path=frame_path,
        )


def capture_frame(robot, camera_key: str):
    observation = robot.get_observation()
    if camera_key not in observation:
        raise RuntimeError(f"Camera key {camera_key!r} not found in robot observation")
    return observation[camera_key]


def estimate_jacobian(
    robot,
    mover: CartesianDeltaMover,
    estimator: LiveFrameEstimator,
    target_key: str,
    camera_key: str,
    output_dir: Path,
    nudge_m: float,
) -> np.ndarray:
    base = estimator.estimate(capture_frame(robot, camera_key), output_dir / "jacobian_base.jpg", target_key)
    columns = []
    for axis, delta in (("x", (nudge_m, 0.0)), ("y", (0.0, nudge_m))):
        mover.move_delta(delta[0], delta[1], duration_s=0.5)
        moved = estimator.estimate(capture_frame(robot, camera_key), output_dir / f"jacobian_{axis}.jpg", target_key)
        columns.append((moved.error_px - base.error_px) / nudge_m)
        mover.move_delta(-delta[0], -delta[1], duration_s=0.5)
    jacobian = np.stack(columns, axis=1)
    if abs(float(np.linalg.det(jacobian))) < 1e-6:
        raise RuntimeError(f"Image-servo Jacobian is singular: {jacobian.tolist()}")
    return jacobian


def run_hover(args: argparse.Namespace) -> dict:
    target_key = normalize_key(args.target_key)
    target_map = load_target_map(args.target_map, required_key=target_key)
    logger.info("Loaded accepted target map: %s", target_map.path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        target = target_map.target_for(target_key)
        return {
            "accepted": True,
            "mode": "dry_run",
            "target_key": target_key,
            "target_px": list(target.center_px),
            "message": "Target map validates; no robot motion executed.",
        }

    if not args.urdf_path:
        raise RuntimeError("--urdf-path is required for robot motion")

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from _so101_waypoints import CartesianDeltaMover, MotionLimits

    estimator = LiveFrameEstimator(args.atakan_code_dir, args.env_file, args.threshold)
    cameras = {
        args.camera_key: OpenCVCameraConfig(
            index_or_path=args.camera_index,
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
        )
    }
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            cameras=cameras,
            max_relative_target=args.max_relative_target,
            use_degrees=True,
        )
    )
    robot.connect()
    try:
        mover = CartesianDeltaMover(
            robot=robot,
            urdf_path=args.urdf_path,
            limits=MotionLimits(
                fps=args.motion_fps,
                max_cartesian_step_m=args.max_cartesian_step_m,
                max_total_correction_m=args.max_total_correction_m,
                max_joint_step_deg=args.max_joint_step_deg,
            ),
        )
        tmp_dir = output_dir / "frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        jacobian = estimate_jacobian(
            robot=robot,
            mover=mover,
            estimator=estimator,
            target_key=target_key,
            camera_key=args.camera_key,
            output_dir=tmp_dir,
            nudge_m=args.jacobian_nudge_m,
        )
        stable = 0
        trace = []
        for iteration in range(args.max_iterations):
            estimate = estimator.estimate(
                capture_frame(robot, args.camera_key),
                tmp_dir / f"servo_{iteration:03d}.jpg",
                target_key,
            )
            error_norm = float(np.linalg.norm(estimate.error_px))
            trace.append(
                {
                    "iteration": iteration,
                    "target_px": estimate.target_px.tolist(),
                    "tip_px": estimate.tip_px.tolist(),
                    "error_px": estimate.error_px.tolist(),
                    "error_norm_px": error_norm,
                }
            )
            if error_norm <= args.pixel_threshold:
                stable += 1
                if stable >= args.stable_frames_required:
                    return {
                        "accepted": True,
                        "mode": "robot_hover",
                        "target_key": target_key,
                        "jacobian": jacobian.tolist(),
                        "trace": trace,
                    }
                continue

            stable = 0
            correction = -np.linalg.pinv(jacobian) @ estimate.error_px
            correction_norm = float(np.linalg.norm(correction))
            if correction_norm > args.max_cartesian_step_m:
                correction = correction * (args.max_cartesian_step_m / correction_norm)
            mover.move_delta(float(correction[0]), float(correction[1]), duration_s=args.motion_duration_s)

        return {
            "accepted": False,
            "mode": "robot_hover",
            "target_key": target_key,
            "reason": "max iterations reached",
            "jacobian": jacobian.tolist(),
            "trace": trace,
        }
    finally:
        if robot.is_connected:
            robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO-101 image-space keyboard hover controller.")
    parser.add_argument("--target-map", required=True)
    parser.add_argument("--target-key", required=True)
    parser.add_argument("--atakan-code-dir", default="../atakan-code")
    parser.add_argument("--env-file", default="../atakan-code/.env.inference")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default="outputs/project4_keyboard_hover")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--robot-port", default="COM_FOLLOWER")
    parser.add_argument("--robot-id", default="so101_keyboard_follower")
    parser.add_argument("--camera-key", default="wrist")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--urdf-path", default=None)

    parser.add_argument("--motion-fps", type=int, default=30)
    parser.add_argument("--motion-duration-s", type=float, default=0.6)
    parser.add_argument("--max-cartesian-step-m", type=float, default=0.012)
    parser.add_argument("--max-total-correction-m", type=float, default=0.060)
    parser.add_argument("--max-joint-step-deg", type=float, default=8.0)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--jacobian-nudge-m", type=float, default=0.005)
    parser.add_argument("--pixel-threshold", type=float, default=15.0)
    parser.add_argument("--stable-frames-required", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        result = run_hover(args)
    except (RuntimeError, TargetMapError, ValueError) as exc:
        result = {"accepted": False, "reason": str(exc)}
        logging.error("%s", exc)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "hover_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {report_path}")
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
