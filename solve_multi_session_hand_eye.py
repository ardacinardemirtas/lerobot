#!/usr/bin/env python
"""
solve_multi_session_hand_eye.py

Joint hand-eye calibration from multiple sessions where the ChArUco board was
repositioned between sessions (but fixed within each session).

Key insight: AX = XB uses RELATIVE motions of the EE and the board-in-camera.
Relative motions cancel out the board's unknown absolute pose, so all
within-session pairs from every session can be pooled into one linear system.

Sessions:
  hand_eye_data      — captures 1-30   (board position 1)
  hand_eye_data_2    — captures 1-45   (board position 2)
  hand_eye_data_3    — captures 1-27   (board position 3)
  hand_eye_data_4    — captures  1-16  (board position 4)
                     — captures 17-21  (board position 5)

Method:
  1. For each session, detect ChArUco board in every image.
  2. Pair with robot pose from sidecar JSON.
  3. Generate all C(N,2) within-session relative-motion pairs (A_ij, B_ij).
  4. Pool all pairs and solve AX = XB via TSAI linear method for rotation,
     then linear least-squares for translation.
  5. Refine with scipy nonlinear least-squares (LM).
  6. Also run per-session OpenCV calibrateHandEye for comparison / sanity check.
  7. Save best result to calibration/hand_eye_result_multi.npz.
"""

import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# --- Paths --------------------------------------------------------------------
CALIB_ROOT = Path(r"C:\Users\plata\robots\lerobot\calibration")

# --- Camera intrinsics --------------------------------------------------------
K = np.array([
    [341.4095,   0.0000, 329.1160],
    [  0.0000, 340.6890, 219.6222],
    [  0.0000,   0.0000,   1.0000],
], dtype=np.float64)

DIST = np.array(
    [0.0799172, -0.15934891, -0.00045079, -0.00048855, 0.11357439],
    dtype=np.float64,
)

# --- ChArUco board (7 cols × 5 rows, square=23.43 mm, marker=17.2 mm) ---------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD      = cv2.aruco.CharucoBoard((7, 5), 0.02343, 0.0172, ARUCO_DICT)
DETECTOR   = cv2.aruco.CharucoDetector(BOARD)

MIN_CORNERS     = 6      # minimum ChArUco corners required
MAX_REPROJ_PX   = 2.0    # skip frames with reprojection error above this
MIN_ROT_DEG     = 5.0    # skip pairs with relative rotation below this


# --- Detection ----------------------------------------------------------------

def detect_board_pose(img_bgr):
    """
    Returns (R 3×3, t (3,), reproj_px) of ChArUco board in camera frame,
    or None if detection fails or reprojection error is too large.
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

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, DIST)
    reproj = float(np.sqrt(np.mean(
        (proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2
    )))
    if reproj > MAX_REPROJ_PX:
        return None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten(), reproj


# --- Session loading ----------------------------------------------------------

def load_session(image_paths, label=""):
    """
    Load valid (R_g2b, t_g2b, R_t2c, t_t2c) tuples for a list of image paths.
    Each image must have a sidecar .json with R_gripper2base / t_gripper2base.
    """
    poses = []
    for img_path in image_paths:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            print(f"  [{img_path.stem}] no JSON — skip")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{img_path.stem}] image load failed — skip")
            continue

        result = detect_board_pose(img)
        if result is None:
            print(f"  [{img_path.stem}] detection failed / high reproj — skip")
            continue

        R_t2c, t_t2c, reproj = result

        pose = json.loads(json_path.read_text())
        R_g2b = np.array(pose["R_gripper2base"], dtype=np.float64)
        t_g2b = np.array(pose["t_gripper2base"], dtype=np.float64).flatten()

        if abs(np.linalg.det(R_g2b) - 1.0) > 0.01:
            print(f"  [{img_path.stem}] bad robot R (det={np.linalg.det(R_g2b):.4f}) — skip")
            continue

        print(f"  [{img_path.stem}] OK  reproj={reproj:.2f}px  "
              f"t_cam=({t_t2c[0]:+.3f},{t_t2c[1]:+.3f},{t_t2c[2]:+.3f})")
        poses.append((R_g2b, t_g2b, R_t2c, t_t2c))

    return poses


# --- Relative motion pairs ----------------------------------------------------

def make_pairs(poses, min_rot_deg=MIN_ROT_DEG):
    """
    Generate all C(N,2) within-session relative motion pairs.

    Convention (matches AX = XB where X = T_ee_cam):
      A_ij = T_g2b_i^{-1} @ T_g2b_j        (relative EE motion in base frame)
      B_ij = T_t2c_i @ T_t2c_j^{-1}        (relative board-in-cam motion)

    Pairs with ||angle(R_A)|| < min_rot_deg are excluded (degenerate for TSAI).
    """
    pairs = []
    for (R_gi, t_gi, R_ci, t_ci), (R_gj, t_gj, R_cj, t_cj) in combinations(poses, 2):
        R_A = R_gi.T @ R_gj
        t_A = R_gi.T @ (t_gj - t_gi)

        R_B = R_ci @ R_cj.T
        t_B = t_ci - R_B @ t_cj

        cos_a = np.clip((np.trace(R_A) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = np.rad2deg(np.arccos(cos_a))
        if angle_deg > min_rot_deg:
            pairs.append((R_A, t_A, R_B, t_B))
    return pairs


# --- TSAI linear solve --------------------------------------------------------

def _mod_rodrigues(R):
    """Rotation matrix → modified Rodrigues vector p = tan(θ/2)*k."""
    rv = Rotation.from_matrix(R).as_rotvec()
    theta = np.linalg.norm(rv)
    if theta < 1e-10:
        return np.zeros(3)
    return np.tan(theta / 2.0) * (rv / theta)


def tsai_rotation(pairs):
    """Solve for R_X in AX=XB (rotation part) via TSAI 1989 linear method."""
    M, v = [], []
    for R_A, _, R_B, _ in pairs:
        p_a = _mod_rodrigues(R_A)
        p_b = _mod_rodrigues(R_B)
        s = p_a + p_b
        skew_s = np.array([
            [    0, -s[2],  s[1]],
            [ s[2],     0, -s[0]],
            [-s[1],  s[0],     0],
        ])
        M.append(skew_s)
        v.append(p_b - p_a)

    p_x, _, _, _ = np.linalg.lstsq(np.vstack(M), np.concatenate(v), rcond=None)

    norm = np.linalg.norm(p_x)
    theta = 2.0 * np.arctan(norm)
    k = p_x / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])
    return Rotation.from_rotvec(theta * k).as_matrix()


def tsai_translation(pairs, R_X):
    """Solve for t_X in AX=XB (translation part) given R_X."""
    # (R_A - I) t_X = R_X t_B - t_A
    A_rows, b_rows = [], []
    for R_A, t_A, R_B, t_B in pairs:
        A_rows.append(R_A - np.eye(3))
        b_rows.append(R_X @ t_B - t_A)

    t_X, _, _, _ = np.linalg.lstsq(
        np.vstack(A_rows), np.concatenate(b_rows), rcond=None
    )
    return t_X


# --- Nonlinear refinement -----------------------------------------------------

def _residuals(params, pairs):
    R_X = Rotation.from_rotvec(params[:3]).as_matrix()
    t_X = params[3:]
    res = []
    for R_A, t_A, R_B, t_B in pairs:
        res.append((R_A @ R_X - R_X @ R_B).ravel())
        res.append((R_A - np.eye(3)) @ t_X - (R_X @ t_B - t_A))
    return np.concatenate(res)


def refine(pairs, R_init, t_init):
    x0 = np.concatenate([Rotation.from_matrix(R_init).as_rotvec(), t_init])
    sol = least_squares(_residuals, x0, args=(pairs,), method="lm", max_nfev=20000)
    R_X = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    t_X = sol.x[3:]
    return R_X, t_X, sol.cost


# --- Per-session OpenCV solve (comparison) -----------------------------------

def opencv_session(poses, label=""):
    """Run all 5 OpenCV calibrateHandEye methods. Returns dict name→T or None."""
    if len(poses) < 4:
        print(f"  {label}: {len(poses)} poses — need ≥4, skipping OpenCV solve")
        return None

    R_g2b = [p[0] for p in poses]
    t_g2b = [p[1].reshape(3, 1) for p in poses]
    R_t2c = [p[2] for p in poses]
    t_t2c = [p[3].reshape(3, 1) for p in poses]

    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    results = {}
    for name, flag in methods.items():
        try:
            R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = t.flatten()
            results[name] = T
            det = np.linalg.det(R)
            print(f"    {name:<12} t=({t[0,0]:+.4f},{t[1,0]:+.4f},{t[2,0]:+.4f})"
                  f"  det(R)={det:.5f}")
        except Exception as exc:
            print(f"    {name:<12} FAILED: {exc}")
    return results


# --- Rotation averaging (Markley 2007 quaternion eigenvalue method) -----------

def rotation_average(Rs, weights=None):
    if weights is None:
        weights = np.ones(len(Rs))
    weights = np.asarray(weights, dtype=float) / sum(weights)
    M = np.zeros((4, 4))
    for R, w in zip(Rs, weights):
        q = Rotation.from_matrix(R).as_quat()   # [x,y,z,w]
        M += w * np.outer(q, q)
    _, vecs = np.linalg.eigh(M)
    return Rotation.from_quat(vecs[:, -1]).as_matrix()


# --- Pretty printing ----------------------------------------------------------

def print_T(T, label="T_ee_cam"):
    R, t = T[:3, :3], T[:3, 3]
    print(f"\n{label}:")
    print(f"  t = [{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}] m")
    print(f"  R =")
    for row in R:
        print(f"      [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")
    print(f"  det(R) = {np.linalg.det(R):.7f}")


def paste_snippet(R, t):
    print("\n_R_cam_in_ee = np.array([")
    for row in R:
        print(f"    [{row[0]:+.8f}, {row[1]:+.8f}, {row[2]:+.8f}],")
    print("], dtype=float)")
    print(f"_t_cam_in_ee = np.array(["
          f"{t[0]:+.8f}, {t[1]:+.8f}, {t[2]:+.8f}], dtype=float)")


# --- Main ---------------------------------------------------------------------

def main():
    # -- Define sessions --------------------------------------------------------
    sessions = []

    for name in ["hand_eye_data", "hand_eye_data_2", "hand_eye_data_3"]:
        d = CALIB_ROOT / name
        if not d.exists():
            print(f"[skip] {d} not found")
            continue
        imgs = sorted(d.glob("calib_*.png"))
        if imgs:
            sessions.append((name, imgs))

    # hand_eye_data_4: split at capture index 14
    d4 = CALIB_ROOT / "hand_eye_data_4"
    if d4.exists():
        imgs4 = sorted(d4.glob("calib_*.png"))
        get_idx = lambda p: int(p.stem.split("_")[1])
        imgs4a  = [p for p in imgs4 if get_idx(p) <= 14]
        imgs4b  = [p for p in imgs4 if get_idx(p) > 14]
        if imgs4a:
            sessions.append(("hand_eye_data_4 [1-14]",  imgs4a))
        if imgs4b:
            sessions.append(("hand_eye_data_4 [15-21]", imgs4b))

    print(f"Sessions ({len(sessions)} total):")
    for lbl, imgs in sessions:
        print(f"  {lbl}: {len(imgs)} images")
    print()

    # -- Process each session ---------------------------------------------------
    all_pairs     = []
    session_poses = {}

    for label, imgs in sessions:
        print(f"{'-'*62}")
        print(f"Session: {label}")
        poses = load_session(imgs, label)
        print(f"  → {len(poses)} valid detections / {len(imgs)} images")
        session_poses[label] = poses

        pairs = make_pairs(poses)
        print(f"  → {len(pairs)} motion pairs (rot ≥ {MIN_ROT_DEG}°)")
        all_pairs.extend(pairs)

    print(f"\n{'='*62}")
    print(f"Total pooled pairs: {len(all_pairs)}")

    if len(all_pairs) < 4:
        print("[error] Not enough pairs — check board detection / lighting.")
        return

    # -- Joint solve: TSAI linear -----------------------------------------------
    print(f"\n{'-'*62}")
    print("Joint solve — TSAI linear init:")
    R_tsai = tsai_rotation(all_pairs)
    t_tsai = tsai_translation(all_pairs, R_tsai)
    print(f"  R det={np.linalg.det(R_tsai):.6f}")
    print(f"  t = ({t_tsai[0]:+.4f}, {t_tsai[1]:+.4f}, {t_tsai[2]:+.4f}) m")

    # -- Joint solve: nonlinear refinement --------------------------------------
    print("\nJoint solve — nonlinear refinement (LM):")
    R_nls, t_nls, cost = refine(all_pairs, R_tsai, t_tsai)
    print(f"  R det={np.linalg.det(R_nls):.6f}  residual cost={cost:.6f}")
    print(f"  t = ({t_nls[0]:+.4f}, {t_nls[1]:+.4f}, {t_nls[2]:+.4f}) m")

    T_joint = np.eye(4)
    T_joint[:3, :3] = R_nls
    T_joint[:3,  3] = t_nls

    # -- Per-session OpenCV results (sanity check) ------------------------------
    print(f"\n{'-'*62}")
    print("Per-session OpenCV calibrateHandEye (sanity check):")
    cv_Rs, cv_ts, cv_ws = [], [], []

    for label, poses in session_poses.items():
        print(f"\n  [{label}]  ({len(poses)} poses)")
        results = opencv_session(poses, label)
        if results:
            best = next(
                (n for n in ("PARK", "HORAUD", "DANIILIDIS", "TSAI", "ANDREFF")
                 if n in results), None
            )
            if best:
                T = results[best]
                cv_Rs.append(T[:3, :3])
                cv_ts.append(T[:3,  3])
                cv_ws.append(len(poses))

    if cv_Rs:
        R_avg = rotation_average(cv_Rs, cv_ws)
        t_avg = np.average(cv_ts, axis=0, weights=cv_ws)
        T_avg = np.eye(4)
        T_avg[:3, :3] = R_avg
        T_avg[:3,  3] = t_avg
        print_T(T_avg, "Per-session weighted average (OpenCV PARK method)")

    # -- Final result -----------------------------------------------------------
    print(f"\n{'='*62}")
    print_T(T_joint, "FINAL: Joint solve (TSAI init + NLS, all sessions pooled)")

    out_path = CALIB_ROOT / "hand_eye_result_multi.npz"
    np.savez(
        str(out_path),
        R_ee_cam=R_nls,
        t_ee_cam=t_nls,
        T_ee_cam=T_joint,
    )
    print(f"\nSaved → {out_path}")

    print(f"\n{'='*62}")
    print("Paste into click_to_move.py")
    print(f"{'='*62}")
    paste_snippet(R_nls, t_nls)


if __name__ == "__main__":
    main()
