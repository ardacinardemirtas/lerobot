#!/usr/bin/env python
"""
Calculated keyboard PnP simulation.

This is not a physics simulation. It places the known keyboard layout at a
synthetic camera pose, projects key centers through OpenCV camera intrinsics,
creates Roboflow-shaped fake detections with noise/dropouts, and runs the real
keyboard_pnp pipeline on those detections.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from keyboard_pnp import (
    LAYOUT,
    _normalize,
    detections_from_roboflow_accumulated,
    get_key_positions_in_base_frame,
    reset_keyboard_cache,
)


DEFAULT_CAMERA_K = np.array(
    [
        [341.4095, 0.0000, 329.1160],
        [0.0000, 340.6890, 219.6222],
        [0.0000, 0.0000, 1.0000],
    ],
    dtype=float,
)
DEFAULT_DIST_COEFFS = np.array(
    [0.0799172, -0.15934891, -0.00045079, -0.00048855, 0.11357439],
    dtype=float,
)
IMAGE_SIZE = (640, 480)


PredictionBatch = list[dict]


@dataclass
class SimulationResult:
    target_key: str
    reproj_err_px: float
    target_error_mm: float
    unique_layout_keys: int
    layout_keys_per_frame: list[int]
    prediction_batches: list[PredictionBatch]
    detections: dict[str, tuple[float, float]]
    projected_pixels: dict[str, tuple[float, float]]
    estimated_positions: dict[str, np.ndarray]
    ground_truth_positions: dict[str, np.ndarray]
    T_base_cam: np.ndarray


def keyboard_object_point(label: str) -> np.ndarray:
    x_m, y_m = LAYOUT[label]
    return np.array([x_m, 0.0, y_m], dtype=float)


def keyboard_object_center() -> np.ndarray:
    xs = [point[0] for point in LAYOUT.values()]
    ys = [point[1] for point in LAYOUT.values()]
    return np.array(
        [
            (min(xs) + max(xs)) / 2.0,
            0.0,
            (min(ys) + max(ys)) / 2.0,
        ],
        dtype=float,
    )


def default_keyboard_pose(depth_m: float = 0.38) -> np.ndarray:
    """
    Return T_base_keyboard for a fronto-parallel keyboard in the camera frame.

    Base and camera frames are identical in this calculated simulation. The
    keyboard object frame uses x=key columns, y=surface normal, z=key rows.
    """
    R_cam_keyboard = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    t_cam_keyboard = np.array([0.0, 0.0, depth_m], dtype=float)
    t_cam_keyboard -= R_cam_keyboard @ keyboard_object_center()

    T_base_keyboard = np.eye(4, dtype=float)
    T_base_keyboard[:3, :3] = R_cam_keyboard
    T_base_keyboard[:3, 3] = t_cam_keyboard
    return T_base_keyboard


def project_key_centers(
    T_base_keyboard: np.ndarray,
    T_base_cam: np.ndarray,
    camera_k: np.ndarray = DEFAULT_CAMERA_K,
    dist_coeffs: np.ndarray = DEFAULT_DIST_COEFFS,
) -> dict[str, tuple[float, float]]:
    T_cam_keyboard = np.linalg.inv(T_base_cam) @ T_base_keyboard
    rvec, _ = cv2.Rodrigues(T_cam_keyboard[:3, :3])
    tvec = T_cam_keyboard[:3, 3]

    labels = list(LAYOUT)
    obj = np.array([keyboard_object_point(label) for label in labels], dtype=float)
    pixels, _ = cv2.projectPoints(obj, rvec, tvec, camera_k, dist_coeffs)
    pixels = pixels.reshape(-1, 2)
    return {
        label: (float(pixel[0]), float(pixel[1]))
        for label, pixel in zip(labels, pixels, strict=True)
    }


def ground_truth_key_positions(T_base_keyboard: np.ndarray) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for label in LAYOUT:
        p_keyboard = np.append(keyboard_object_point(label), 1.0)
        positions[label] = (T_base_keyboard @ p_keyboard)[:3].copy()
    return positions


def visible_pixels(
    projected_pixels: dict[str, tuple[float, float]],
    image_size: tuple[int, int] = IMAGE_SIZE,
    margin_px: float = 20.0,
) -> dict[str, tuple[float, float]]:
    width, height = image_size
    return {
        label: (u, v)
        for label, (u, v) in projected_pixels.items()
        if -margin_px <= u <= width + margin_px and -margin_px <= v <= height + margin_px
    }


def make_prediction_batches(
    projected_pixels: dict[str, tuple[float, float]],
    frames: int,
    keys_per_frame: int,
    noise_px: float,
    rng: np.random.Generator,
    drop_rate: float = 0.0,
    duplicate_rate: float = 0.0,
    include_keyboard_box: bool = True,
) -> list[PredictionBatch]:
    labels = list(projected_pixels)
    rng.shuffle(labels)
    cursor = 0
    batches: list[PredictionBatch] = []

    all_pixels = np.array(list(projected_pixels.values()), dtype=float)
    x_min, y_min = np.min(all_pixels, axis=0)
    x_max, y_max = np.max(all_pixels, axis=0)

    for _ in range(frames):
        if cursor + keys_per_frame > len(labels):
            rng.shuffle(labels)
            cursor = 0
        chosen = labels[cursor : cursor + keys_per_frame]
        cursor += keys_per_frame

        preds: PredictionBatch = []
        if include_keyboard_box:
            preds.append(
                {
                    "class": "keyboard",
                    "x": float((x_min + x_max) / 2.0),
                    "y": float((y_min + y_max) / 2.0),
                    "width": float(x_max - x_min),
                    "height": float(y_max - y_min),
                    "confidence": 0.99,
                }
            )

        for label in chosen:
            if rng.random() < drop_rate:
                continue
            u, v = projected_pixels[label]
            noisy_u = float(u + rng.normal(0.0, noise_px))
            noisy_v = float(v + rng.normal(0.0, noise_px))
            confidence = float(rng.uniform(0.72, 0.98))
            preds.append(
                {
                    "class": label,
                    "x": noisy_u,
                    "y": noisy_v,
                    "width": 18.0,
                    "height": 16.0,
                    "confidence": confidence,
                }
            )
            if rng.random() < duplicate_rate:
                preds.append(
                    {
                        "class": label,
                        "x": float(noisy_u + rng.normal(0.0, noise_px * 2.0)),
                        "y": float(noisy_v + rng.normal(0.0, noise_px * 2.0)),
                        "width": 18.0,
                        "height": 16.0,
                        "confidence": float(confidence * 0.65),
                    }
                )
        batches.append(preds)

    return batches


def count_layout_keys_per_frame(prediction_batches: list[PredictionBatch]) -> list[int]:
    counts: list[int] = []
    for preds in prediction_batches:
        labels = {
            _normalize(pred["class"])
            for pred in preds
            if pred["class"].lower() != "keyboard" and _normalize(pred["class"]) in LAYOUT
        }
        counts.append(len(labels))
    return counts


def run_simulation(
    target_key: str = "a",
    frames: int = 5,
    keys_per_frame: int = 3,
    noise_px: float = 1.0,
    seed: int = 0,
    drop_rate: float = 0.0,
    duplicate_rate: float = 0.0,
    min_hits: int = 1,
) -> SimulationResult:
    target_key = _normalize(target_key)
    if target_key not in LAYOUT:
        raise ValueError(f"Unsupported target key {target_key!r}")

    rng = np.random.default_rng(seed)
    T_base_cam = np.eye(4, dtype=float)
    T_base_keyboard = default_keyboard_pose()
    projected = project_key_centers(T_base_keyboard, T_base_cam)
    visible = visible_pixels(projected)
    batches = make_prediction_batches(
        visible,
        frames=frames,
        keys_per_frame=keys_per_frame,
        noise_px=noise_px,
        rng=rng,
        drop_rate=drop_rate,
        duplicate_rate=duplicate_rate,
    )
    detections = detections_from_roboflow_accumulated(batches, min_hits=min_hits)

    reset_keyboard_cache()
    estimated_positions, reproj_err = get_key_positions_in_base_frame(
        detections,
        T_base_cam,
        DEFAULT_CAMERA_K,
        DEFAULT_DIST_COEFFS,
    )
    gt_positions = ground_truth_key_positions(T_base_keyboard)
    target_error_mm = float(
        np.linalg.norm(estimated_positions[target_key] - gt_positions[target_key]) * 1000.0
    )

    return SimulationResult(
        target_key=target_key,
        reproj_err_px=float(reproj_err),
        target_error_mm=target_error_mm,
        unique_layout_keys=len(detections),
        layout_keys_per_frame=count_layout_keys_per_frame(batches),
        prediction_batches=batches,
        detections=detections,
        projected_pixels=projected,
        estimated_positions=estimated_positions,
        ground_truth_positions=gt_positions,
        T_base_cam=T_base_cam,
    )


def project_base_point(
    point_base: np.ndarray,
    T_base_cam: np.ndarray,
    camera_k: np.ndarray = DEFAULT_CAMERA_K,
    dist_coeffs: np.ndarray = DEFAULT_DIST_COEFFS,
) -> tuple[int, int] | None:
    T_cam_base = np.linalg.inv(T_base_cam)
    point_cam = (T_cam_base @ np.append(point_base, 1.0))[:3]
    if point_cam[2] <= 0:
        return None
    pixels, _ = cv2.projectPoints(
        point_cam.reshape(1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        camera_k,
        dist_coeffs,
    )
    u, v = pixels.reshape(2)
    return int(round(u)), int(round(v))


def render_overlay(
    result: SimulationResult,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    width, height = image_size
    image = np.full((height, width, 3), (238, 238, 238), dtype=np.uint8)

    for label, (u, v) in result.projected_pixels.items():
        p = (int(round(u)), int(round(v)))
        cv2.circle(image, p, 2, (170, 170, 170), -1)
        if label in {"a", "s", "d", "f", "j", "k", "l", "space", "enter"}:
            cv2.putText(image, label, (p[0] + 3, p[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)

    colors = [(255, 120, 0), (230, 80, 0), (200, 40, 0), (170, 0, 0), (130, 0, 0)]
    for frame_idx, preds in enumerate(result.prediction_batches):
        color = colors[frame_idx % len(colors)]
        for pred in preds:
            if pred["class"].lower() == "keyboard":
                continue
            p = (int(round(pred["x"])), int(round(pred["y"])))
            cv2.circle(image, p, 5, color, 1)

    gt_px = project_base_point(
        result.ground_truth_positions[result.target_key],
        result.T_base_cam,
    )
    est_px = project_base_point(
        result.estimated_positions[result.target_key],
        result.T_base_cam,
    )
    if gt_px is not None:
        cv2.drawMarker(image, gt_px, (0, 170, 0), cv2.MARKER_CROSS, 18, 2)
    if est_px is not None:
        cv2.drawMarker(image, est_px, (0, 0, 230), cv2.MARKER_TILTED_CROSS, 18, 2)

    cv2.putText(
        image,
        f"target={result.target_key} err={result.target_error_mm:.2f}mm reproj={result.reproj_err_px:.2f}px",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"layout keys={result.unique_layout_keys} per-frame={result.layout_keys_per_frame}",
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return image


def save_overlay(path: str | Path, result: SimulationResult) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), render_overlay(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calculated keyboard PnP simulation.")
    parser.add_argument("--key", default="a", help="Target key to evaluate.")
    parser.add_argument("--frames", type=int, default=5, help="Detection frames to accumulate.")
    parser.add_argument("--keys-per-frame", type=int, default=3, help="Visible keys per simulated frame.")
    parser.add_argument("--noise-px", type=float, default=1.0, help="Gaussian pixel noise for fake detections.")
    parser.add_argument("--drop-rate", type=float, default=0.0, help="Probability of dropping a chosen key.")
    parser.add_argument("--duplicate-rate", type=float, default=0.0, help="Probability of duplicate key detections.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlay-out", default="outputs/sim_keyboard_pnp_overlay.png")
    parser.add_argument("--save-overlay", action="store_true")
    args = parser.parse_args()

    result = run_simulation(
        target_key=args.key,
        frames=args.frames,
        keys_per_frame=args.keys_per_frame,
        noise_px=args.noise_px,
        seed=args.seed,
        drop_rate=args.drop_rate,
        duplicate_rate=args.duplicate_rate,
    )

    print(f"target key         : {result.target_key}")
    print(f"frames             : {args.frames}")
    print(f"layout keys/frame  : {result.layout_keys_per_frame}")
    print(f"unique layout keys : {result.unique_layout_keys}")
    print(f"PnP reproj error   : {result.reproj_err_px:.3f} px")
    print(f"target error       : {result.target_error_mm:.3f} mm")

    if args.save_overlay:
        save_overlay(args.overlay_out, result)
        print(f"wrote overlay      : {args.overlay_out}")


if __name__ == "__main__":
    main()
