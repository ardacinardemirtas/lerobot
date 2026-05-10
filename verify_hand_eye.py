#!/usr/bin/env python
"""
verify_hand_eye.py  --  sanity-check the hand-eye calibration result.

Three independent tests:

  1. CORNER-ID MAP
     Each detected ChArUco corner is labelled with its board ID.
     If detection is correct, corner #0 must always sit at the same
     physical square of the board across every image.

  2. BOARD-POSITION CONSISTENCY  (works when the board is stationary)
     T_base_board = T_base_ee  *  T_ee_cam  *  T_cam_board
     If T_ee_cam is correct, the computed board origin in the robot
     base frame should be the same for every capture.
     We report mean and std-dev of all 30 estimates.

  3. REPROJECTION VIA T_ee_cam  (hardest test)
     For each frame we independently derive where the board should be
     in the camera using only the robot FK pose + T_ee_cam:
         T_cam_board_pred = inv(T_ee_cam) * inv(T_base_ee) * T_base_board_mean
     We then project the board's 3D corners onto the image and draw them
     in a different colour from the *detected* corners.
     Close overlap means the calibration is geometrically consistent.

Outputs  (written to <DATA_DIR>/verification/):
   corner_ids_NNNN.jpg   -- annotated image with corner IDs
   reproj_NNNN.jpg       -- detected (green) vs predicted (red) corners
   summary_grid.jpg      -- first 9 corner-id images tiled in a 3x3 grid
   consistency.txt       -- numerical consistency report
"""

import json
from pathlib import Path

import cv2
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\plata\robots\lerobot\calibration\hand_eye_data")
OUT_DIR  = DATA_DIR / "verification"

K    = np.array([[341.4095, 0, 329.1160],
                 [0, 340.6890, 219.6222],
                 [0, 0, 1]], dtype=np.float64)
DIST = np.array([0.0799172, -0.15934891, -0.00045079, -0.00048855, 0.11357439],
                dtype=np.float64)

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD      = cv2.aruco.CharucoBoard((7, 5), 0.02343, 0.0172, ARUCO_DICT)
DETECTOR   = cv2.aruco.CharucoDetector(BOARD)

MIN_CORNERS = 6

# ── helpers ────────────────────────────────────────────────────────────────────

def detect(img_bgr):
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
    R, _ = cv2.Rodrigues(rvec)
    return dict(corners=corners, ids=ids, R=R, t=tvec.reshape(3),
                rvec=rvec, tvec=tvec, obj_pts=obj_pts, img_pts=img_pts)


def T4(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3,  3] = t.flatten()
    return M


def inv4(M):
    R, t = M[:3, :3], M[:3, 3]
    Mi = np.eye(4)
    Mi[:3, :3] = R.T
    Mi[:3,  3] = -R.T @ t
    return Mi


def project(pts_3d, rvec, tvec):
    proj, _ = cv2.projectPoints(pts_3d.reshape(-1, 1, 3), rvec, tvec, K, DIST)
    return proj.reshape(-1, 2)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load hand-eye result
    npz_path = DATA_DIR / "hand_eye_result.npz"
    if not npz_path.exists():
        print(f"[error] {npz_path} not found -- run solve_charuco_hand_eye.py first")
        return
    he = np.load(str(npz_path))
    T_ee_cam = he["T_ee_cam"]          # camera in EE frame
    T_cam_ee = inv4(T_ee_cam)          # EE in camera frame

    images   = sorted(DATA_DIR.glob("calib_*.png"))
    frames   = []   # list of dicts with all per-frame data

    # ── detect all frames ───────────────────────────────────────────────────────
    print(f"Detecting ChArUco in {len(images)} images ...")
    for img_path in images:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        pose = json.loads(json_path.read_text())
        R_ee = np.array(pose["R_gripper2base"], dtype=np.float64)
        t_ee = np.array(pose["t_gripper2base"], dtype=np.float64)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        d = detect(img)
        if d is None:
            print(f"  [{img_path.stem}] detection failed -- skipped")
            continue

        frames.append(dict(stem=img_path.stem, img=img,
                           R_ee=R_ee, t_ee=t_ee, **d))

    print(f"OK: {len(frames)} / {len(images)}\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 1 -- Corner ID map
    # ═══════════════════════════════════════════════════════════════════════════
    print("Test 1: Corner-ID maps ...")
    grid_imgs = []
    for f in frames:
        vis = f["img"].copy()

        # draw each detected corner with its charuco ID
        for pt, cid in zip(f["corners"].reshape(-1, 2), f["ids"].flatten()):
            x, y = int(pt[0]), int(pt[1])
            # filled circle
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            # ID label (white text, black outline for readability)
            label = str(int(cid))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(vis, (x+7, y-th-3), (x+7+tw+2, y+2), (0,0,0), -1)
            cv2.putText(vis, label, (x+8, y-1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # number of detected corners in top-left
        cv2.putText(vis, f"{f['stem']}  n={len(f['ids'])}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2, cv2.LINE_AA)

        out = OUT_DIR / f"corner_ids_{f['stem']}.jpg"
        cv2.imwrite(str(out), vis)
        grid_imgs.append(vis)

    # 3x3 summary grid
    n_grid = min(9, len(grid_imgs))
    cols   = 3
    rows   = (n_grid + cols - 1) // cols
    h, w   = grid_imgs[0].shape[:2]
    th, tw = h // 2, w // 2          # thumbnail size
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, gi in enumerate(grid_imgs[:n_grid]):
        r, c = divmod(i, cols)
        thumb = cv2.resize(gi, (tw, th))
        canvas[r*th:(r+1)*th, c*tw:(c+1)*tw] = thumb
    cv2.imwrite(str(OUT_DIR / "summary_grid.jpg"), canvas)
    print(f"  Saved corner-ID images to {OUT_DIR}/corner_ids_*.jpg")
    print(f"  Saved summary grid       to {OUT_DIR}/summary_grid.jpg\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 2 -- Board position consistency
    # ═══════════════════════════════════════════════════════════════════════════
    print("Test 2: Board position consistency in robot base frame ...")
    board_origins = []   # board origin expressed in robot base frame
    board_xaxes   = []
    for f in frames:
        T_base_ee  = T4(f["R_ee"], f["t_ee"])
        T_cam_board = T4(f["R"], f["t"])
        T_base_board = T_base_ee @ T_ee_cam @ T_cam_board
        board_origins.append(T_base_board[:3, 3])
        board_xaxes.append(T_base_board[:3, 0])   # first column = board X axis in base frame

    origins = np.array(board_origins)   # (N, 3)
    mean_o  = origins.mean(axis=0)
    std_o   = origins.std(axis=0)
    rms_o   = float(np.sqrt(np.mean(np.sum((origins - mean_o)**2, axis=1))))

    print(f"  Board origin in base frame (should be constant if board was fixed)")
    print(f"  Mean  : x={mean_o[0]:+.4f}  y={mean_o[1]:+.4f}  z={mean_o[2]:+.4f}  m")
    print(f"  StdDev: x={std_o[0]:+.4f}  y={std_o[1]:+.4f}  z={std_o[2]:+.4f}  m")
    print(f"  RMS deviation from mean: {rms_o*100:.2f} cm\n")

    per_frame_lines = []
    for i, (f, o) in enumerate(zip(frames, origins)):
        dev = np.linalg.norm(o - mean_o) * 100
        per_frame_lines.append(f"  [{f['stem']}]  board_origin=({o[0]:+.4f},{o[1]:+.4f},{o[2]:+.4f})m  dev={dev:.2f}cm")
        print(per_frame_lines[-1])

    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 3 -- Reprojection via T_ee_cam
    # ═══════════════════════════════════════════════════════════════════════════
    print("\nTest 3: Reprojection via T_ee_cam (detected=green, predicted=red) ...")

    # use the mean board pose in base frame as the "ground truth" board location
    mean_R_board = np.array(board_xaxes).mean(axis=0)   # rough mean orientation
    # build a mean T_base_board from per-frame values
    T_base_boards = []
    for f in frames:
        T_base_ee   = T4(f["R_ee"], f["t_ee"])
        T_cam_board = T4(f["R"], f["t"])
        T_base_boards.append(T_base_ee @ T_ee_cam @ T_cam_board)
    T_base_board_mean = np.mean(T_base_boards, axis=0)
    # re-orthogonalise rotation part via SVD
    U, _, Vt = np.linalg.svd(T_base_board_mean[:3, :3])
    T_base_board_mean[:3, :3] = U @ Vt

    # all 3D board corner positions in world (board) frame
    all_obj_pts = BOARD.getChessboardCorners()   # (N_corners, 3)

    reproj_errors = []
    for f in frames:
        T_base_ee = T4(f["R_ee"], f["t_ee"])
        # predicted: board->base->ee->cam
        T_cam_board_pred = inv4(T_ee_cam) @ inv4(T_base_ee) @ T_base_board_mean
        R_pred = T_cam_board_pred[:3, :3]
        t_pred = T_cam_board_pred[:3, 3]
        rvec_pred, _ = cv2.Rodrigues(R_pred)
        pts_pred = project(all_obj_pts, rvec_pred, t_pred.reshape(3, 1))

        # detected
        pts_det = f["img_pts"].reshape(-1, 2)

        # per-corner reprojection error (detected corners only)
        obj_sub, img_sub = f["obj_pts"], f["img_pts"].reshape(-1, 2)
        rvec_det, _ = cv2.Rodrigues(f["R"])
        pts_pred_sub = project(obj_sub, rvec_det, f["t"].reshape(3, 1))
        err_det = float(np.sqrt(np.mean((pts_pred_sub - img_sub)**2)))

        # predicted vs detected error (for the detected subset)
        ids_flat = f["ids"].flatten()
        obj_sub_pred = all_obj_pts[ids_flat]
        pts_pp = project(obj_sub_pred, rvec_pred, t_pred.reshape(3, 1))
        err_pred = float(np.sqrt(np.mean((pts_pp - img_sub)**2)))

        reproj_errors.append((f["stem"], err_det, err_pred))
        print(f"  [{f['stem']}]  reproj_detect={err_det:.3f}px  reproj_predict={err_pred:.3f}px")

        vis = f["img"].copy()
        # draw ALL predicted board corners (red open circles)
        for pt in pts_pred:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 200), 1)
        # draw detected corners (green filled)
        for pt in img_sub:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 220, 0), -1)
        # legend
        cv2.circle(vis, (12, 15), 6, (0, 220, 0), -1)
        cv2.putText(vis, "detected", (22, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1)
        cv2.circle(vis, (12, 35), 6, (0, 0, 200), 1)
        cv2.putText(vis, "predicted via T_ee_cam", (22, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,200), 1)
        cv2.putText(vis, f"pred err={err_pred:.3f}px", (6, vis.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imwrite(str(OUT_DIR / f"reproj_{f['stem']}.jpg"), vis)

    mean_pred_err = np.mean([e[2] for e in reproj_errors])
    print(f"\n  Mean prediction error (via T_ee_cam): {mean_pred_err:.3f} px")

    # ── write text report ─────────────────────────────────────────────────────
    report = [
        "Hand-Eye Calibration Verification Report",
        "=" * 60,
        "",
        f"Loaded T_ee_cam from: {npz_path}",
        "",
        "T_ee_cam (4x4):",
    ]
    for row in T_ee_cam:
        report.append(f"  {row}")
    report += [
        "",
        "=" * 60,
        "TEST 2 -- Board position consistency",
        "=" * 60,
        f"  Mean board origin (base frame): ({mean_o[0]:+.4f}, {mean_o[1]:+.4f}, {mean_o[2]:+.4f}) m",
        f"  StdDev:                         ({std_o[0]:+.4f}, {std_o[1]:+.4f}, {std_o[2]:+.4f}) m",
        f"  RMS deviation:                  {rms_o*100:.2f} cm",
        "",
    ] + per_frame_lines + [
        "",
        "=" * 60,
        "TEST 3 -- Reprojection via T_ee_cam",
        "=" * 60,
        f"  Mean prediction reprojection error: {mean_pred_err:.3f} px",
        "",
    ] + [f"  [{s}]  detect={d:.3f}px  predict={p:.3f}px" for s, d, p in reproj_errors]

    (OUT_DIR / "consistency.txt").write_text("\n".join(report))
    print(f"\nFull report saved to {OUT_DIR / 'consistency.txt'}")
    print(f"Reprojection images  -> {OUT_DIR}/reproj_*.jpg")


if __name__ == "__main__":
    main()
