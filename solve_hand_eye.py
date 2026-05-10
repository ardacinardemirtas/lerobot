#!/usr/bin/env python
"""
solve_hand_eye.py — Compute T_ee_cam from data collected by calibrate_hand_eye.py.

Usage:
    python solve_hand_eye.py [--data <hand_eye_data_dir>] [--board checkerboard|aruco]

The script:
  1. Loads poses.npz   → R_gripper2base, t_gripper2base
  2. Detects calibration target in each calib_NNNN.png
  3. Runs cv2.calibrateHandEye (TSAI method by default)
  4. Prints the resulting T_ee_cam and saves it to hand_eye_result.npz

Supported calibration targets:
  checkerboard   default 9×6 inner corners, 25 mm squares
  aruco          4×4 ArUco board (cv2.aruco)

Edit the BOARD_* constants below to match your physical target.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ─── Camera intrinsics (same as calibrate_hand_eye.py / click_to_move.py) ────
CAMERA_K = np.array([
    [341.4095,   0.0000, 329.1160],
    [  0.0000, 340.6890, 219.6222],
    [  0.0000,   0.0000,   1.0000],
], dtype=np.float64)

DIST_COEFFS = np.array(
    [0.0799172, -0.15934891, -0.00045079, -0.00048855, 0.11357439],
    dtype=np.float64,
)

# ─── Checkerboard target config ───────────────────────────────────────────────
CB_COLS       = 9      # inner corners along width
CB_ROWS       = 6      # inner corners along height
CB_SQUARE_M   = 0.025  # square size in metres

# ─── ArUco board config ───────────────────────────────────────────────────────
ARUCO_DICT_ID  = cv2.aruco.DICT_4X4_50
ARUCO_COLS     = 4
ARUCO_ROWS     = 3
ARUCO_MARKER_M = 0.04   # marker size (metres)
ARUCO_GAP_M    = 0.01   # gap between markers (metres)

# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = Path(r"C:\Users\plata\robots\lerobot\calibration\hand_eye_data")
RESULT_FILE = DATA_DIR / "hand_eye_result.npz"


def _detect_checkerboard(img_bgr: np.ndarray):
    """Return (rvec, tvec) of checkerboard in camera frame, or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (CB_COLS, CB_ROWS), flags)
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    obj_pts = np.zeros((CB_ROWS * CB_COLS, 3), dtype=np.float32)
    obj_pts[:, :2] = np.mgrid[0:CB_COLS, 0:CB_ROWS].T.reshape(-1, 2) * CB_SQUARE_M

    _, rvec, tvec = cv2.solvePnP(obj_pts, corners, CAMERA_K, DIST_COEFFS)
    return rvec, tvec


def _detect_aruco(img_bgr: np.ndarray):
    """Return (rvec, tvec) of ArUco board in camera frame, or None."""
    aruco_dict  = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()
    detector    = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(img_bgr)
    if ids is None or len(ids) < 3:
        return None

    board = cv2.aruco.GridBoard(
        (ARUCO_COLS, ARUCO_ROWS),
        ARUCO_MARKER_M,
        ARUCO_GAP_M,
        aruco_dict,
    )
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None

    _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, CAMERA_K, DIST_COEFFS)
    return rvec, tvec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=str(DATA_DIR))
    parser.add_argument("--board", default="checkerboard", choices=["checkerboard", "aruco"])
    args = parser.parse_args()

    data_dir = Path(args.data)
    npz_path = data_dir / "poses.npz"

    if not npz_path.exists():
        print(f"[error] poses.npz not found in {data_dir}")
        sys.exit(1)

    robot_data = np.load(str(npz_path))
    R_g2b_all  = list(robot_data["R_gripper2base"])   # list of 3×3
    t_g2b_all  = list(robot_data["t_gripper2base"])   # list of (3,)
    n_robot    = len(R_g2b_all)

    detect_fn = _detect_checkerboard if args.board == "checkerboard" else _detect_aruco

    images = sorted(data_dir.glob("calib_*.png"))
    print(f"Found {len(images)} images, {n_robot} robot poses.")

    R_t2c_list, t_t2c_list = [], []
    used_indices            = []

    for img_path in images:
        idx_str = img_path.stem.split("_")[1]   # "0001"
        idx     = int(idx_str)                   # 1-based

        if idx > n_robot:
            print(f"  [{idx_str}] no matching robot pose — skipping")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [{idx_str}] failed to load image — skipping")
            continue

        result = detect_fn(img_bgr)
        if result is None:
            print(f"  [{idx_str}] target not detected — skipping")
            continue

        rvec, tvec = result
        R_t2c, _   = cv2.Rodrigues(rvec)
        R_t2c_list.append(R_t2c)
        t_t2c_list.append(tvec.flatten())
        used_indices.append(idx - 1)   # convert to 0-based
        print(f"  [{idx_str}] OK  t_cam=({tvec[0,0]:+.3f}, {tvec[1,0]:+.3f}, {tvec[2,0]:+.3f})")

    n_used = len(used_indices)
    print(f"\n{n_used} usable pairs (need ≥ 4 for TSAI).")
    if n_used < 4:
        print("[error] Not enough valid pairs. Collect more data or check target detection.")
        sys.exit(1)

    R_g2b = [R_g2b_all[i] for i in used_indices]
    t_g2b = [t_g2b_all[i].reshape(3, 1) for i in used_indices]

    R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c_list,
        t_target2cam=[t.reshape(3, 1) for t in t_t2c_list],
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    T_ee_cam = np.eye(4)
    T_ee_cam[:3, :3] = R_ee_cam
    T_ee_cam[:3,  3] = t_ee_cam.flatten()

    print("\n─── Hand-Eye Result (T_ee_cam) ─────────────────────────────")
    print(f"R_ee_cam =\n{np.array2string(R_ee_cam, precision=6, sign='+')}")
    print(f"t_ee_cam = {np.array2string(t_ee_cam.flatten(), precision=6, sign='+')}")
    print(f"\nT_ee_cam (4×4) =\n{np.array2string(T_ee_cam, precision=6, sign='+')}")

    result_path = data_dir / "hand_eye_result.npz"
    np.savez(str(result_path), R_ee_cam=R_ee_cam, t_ee_cam=t_ee_cam, T_ee_cam=T_ee_cam)
    print(f"\nSaved → {result_path}")

    # Pretty-print snippet to paste into click_to_move.py
    R = R_ee_cam
    t = t_ee_cam.flatten()
    print("\n── Paste into click_to_move.py ──")
    print(f"_R_cam_in_ee = np.array([")
    for row in R:
        print(f"    [{row[0]:+.8f}, {row[1]:+.8f}, {row[2]:+.8f}],")
    print(f"], dtype=float)")
    print(f"_t_cam_in_ee = np.array([{t[0]:+.8f}, {t[1]:+.8f}, {t[2]:+.8f}], dtype=float)")


if __name__ == "__main__":
    main()
