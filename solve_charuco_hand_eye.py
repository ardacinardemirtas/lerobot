#!/usr/bin/env python
"""
solve_charuco_hand_eye.py
Compute T_ee_cam (end-effector → camera) from ChArUco images + robot poses.

Board  : DICT_4X4_50  7 cols × 5 rows
Square : 2.343 cm   Marker: 1.72 cm
"""

import json
from pathlib import Path

import cv2
import numpy as np

# ─── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\plata\robots\lerobot\calibration\hand_eye_data")
VIZ_DIR  = DATA_DIR / "detections"

# ─── camera intrinsics ────────────────────────────────────────────────────────
K = np.array([
    [341.4095,   0.0000, 329.1160],
    [  0.0000, 340.6890, 219.6222],
    [  0.0000,   0.0000,   1.0000],
], dtype=np.float64)

DIST = np.array(
    [0.0799172, -0.15934891, -0.00045079, -0.00048855, 0.11357439],
    dtype=np.float64,
)

# ─── ChArUco board ────────────────────────────────────────────────────────────
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD      = cv2.aruco.CharucoBoard((7, 5), 0.02343, 0.0172, ARUCO_DICT)
DETECTOR   = cv2.aruco.CharucoDetector(BOARD)

MIN_CORNERS = 6   # minimum detected corners to accept a frame


def detect_pose(img_bgr):
    """
    Returns (R_target2cam 3×3, t_target2cam 3×1, n_corners, reproj_px)
    or None if detection fails.
    """
    corners, ids, _, _ = DETECTOR.detectBoard(img_bgr)
    if ids is None or len(ids) < MIN_CORNERS:
        return None

    obj_pts, img_pts = BOARD.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < MIN_CORNERS:
        return None

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, DIST,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    # reprojection error
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, DIST)
    err = float(np.sqrt(np.mean((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2)))

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1), int(len(ids)), err, rvec, corners, ids


def save_viz(img_bgr, corners, ids, rvec, tvec, path):
    vis = img_bgr.copy()
    cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
    cv2.drawFrameAxes(vis, K, DIST, rvec, tvec, 0.03)
    cv2.imwrite(str(path), vis)


def main():
    images = sorted(DATA_DIR.glob("calib_*.png"))
    print(f"Found {len(images)} images in {DATA_DIR}\n")

    VIZ_DIR.mkdir(exist_ok=True)

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []
    skipped       = []

    for img_path in images:
        stem      = img_path.stem
        json_path = img_path.with_suffix(".json")

        if not json_path.exists():
            print(f"  [{stem}]  no JSON  — skip")
            skipped.append(stem)
            continue

        pose  = json.loads(json_path.read_text())
        R_ee  = np.array(pose["R_gripper2base"], dtype=np.float64)   # 3×3
        t_ee  = np.array(pose["t_gripper2base"], dtype=np.float64).reshape(3, 1)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{stem}]  image load failed  — skip")
            skipped.append(stem)
            continue

        result = detect_pose(img)
        if result is None:
            n_raw = 0
            if cv2.aruco.ArucoDetector(ARUCO_DICT).detectMarkers(img)[1] is not None:
                n_raw = len(cv2.aruco.ArucoDetector(ARUCO_DICT).detectMarkers(img)[1])
            print(f"  [{stem}]  ChArUco detection failed (raw markers: {n_raw})  — skip")
            skipped.append(stem)
            continue

        R_cam, t_cam, n_c, err, rvec, corners, ids = result
        print(f"  [{stem}]  {n_c:2d} corners  reproj={err:.3f}px"
              f"  t_cam=({t_cam[0,0]:+.3f},{t_cam[1,0]:+.3f},{t_cam[2,0]:+.3f})m")

        save_viz(img, corners, ids, rvec, t_cam, VIZ_DIR / f"{stem}_det.jpg")

        R_g2b.append(R_ee)
        t_g2b.append(t_ee)
        R_t2c.append(R_cam)
        t_t2c.append(t_cam)

    n = len(R_g2b)
    print(f"\nUsable pairs: {n} / {len(images)}"
          + (f"  (skipped: {', '.join(skipped)})" if skipped else ""))

    if n < 4:
        print("[error] Need ≥ 4 pairs. Check board placement / lighting.")
        return

    # ── run all five methods and compare ──────────────────────────────────────
    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {}
    print()
    for name, flag in methods.items():
        try:
            R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
            T      = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = t.flatten()
            det    = np.linalg.det(R)
            results[name] = T
            print(f"  {name:<12}  t=({t[0,0]:+.4f},{t[1,0]:+.4f},{t[2,0]:+.4f})m"
                  f"  det(R)={det:.5f}")
        except Exception as exc:
            print(f"  {name:<12}  FAILED: {exc}")

    if not results:
        print("[error] All methods failed.")
        return

    # PARK = HORAUD = DANIILIDIS are typically consistent; prefer PARK over TSAI
    # when TSAI disagrees with the majority.
    best_name = next((n for n in ("PARK", "HORAUD", "DANIILIDIS", "TSAI", "ANDREFF")
                      if n in results), next(iter(results)))
    T_ee_cam  = results[best_name]
    R_ee_cam  = T_ee_cam[:3, :3]
    t_ee_cam  = T_ee_cam[:3,  3]

    # ── print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Result ({best_name})  -  T_ee_cam  (camera in EE frame)")
    print(f"{'='*62}")
    print(f"  R_ee_cam =")
    for row in R_ee_cam:
        print(f"    [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")
    print(f"  t_ee_cam = [{t_ee_cam[0]:+.6f}, {t_ee_cam[1]:+.6f}, {t_ee_cam[2]:+.6f}] m")
    print(f"\n  T_ee_cam (4×4) =")
    for row in T_ee_cam:
        print(f"    [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}, {row[3]:+.6f}]")

    # ── save ──────────────────────────────────────────────────────────────────
    result_path = DATA_DIR / "hand_eye_result.npz"
    np.savez(str(result_path),
             R_ee_cam=R_ee_cam,
             t_ee_cam=t_ee_cam,
             T_ee_cam=T_ee_cam)
    print(f"\n  Saved → {result_path}")
    print(f"  Detection overlays → {VIZ_DIR}/")

    # ── paste snippet ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Paste into click_to_move.py")
    print(f"{'='*62}")
    print(f"_R_cam_in_ee = np.array([")
    for row in R_ee_cam:
        print(f"    [{row[0]:+.8f}, {row[1]:+.8f}, {row[2]:+.8f}],")
    print(f"], dtype=float)")
    print(f"_t_cam_in_ee = np.array(["
          f"{t_ee_cam[0]:+.8f}, {t_ee_cam[1]:+.8f}, {t_ee_cam[2]:+.8f}], dtype=float)")


if __name__ == "__main__":
    main()
