"""
keyboard_pnp.py — PnP-based 3D keyboard key localization for the SO-101 robot.

Pipeline
--------
1. Detector returns {label: (u, v)} pixel centroids for visible keys.
2. Each label is matched against the known German ISO (QWERTZ) layout to get
   a 3D object point (x_m, y_m, 0.0) in keyboard frame (metres, z=0 for all).
3. cv2.solvePnPRansac finds keyboard-frame → camera-frame transform (R, t).
4. T_base_cam (from FK + hand-eye) maps camera frame → robot base frame.
5. Every key in the layout is projected to robot base coordinates.

Why z=0 is correct and sufficient
----------------------------------
All keys are coplanar, so z=0 in the keyboard frame for every point.  PnP
solves for R and t that satisfy p_cam = R @ p_obj + t.  If you shift all
object z by some constant Δ, PnP absorbs it into t' = t – R·[0,0,Δ]ᵀ and
the actual camera-frame positions are unchanged.  The keyboard's depth from
the camera is recovered automatically from the 2D–3D correspondences — you
never need to specify it.

Usage
-----
    from keyboard_pnp import detections_from_roboflow, get_key_positions_in_base_frame

    preds = detect_keys(frame)                       # from press_key.py
    dets  = detections_from_roboflow(preds)
    T_bc  = kin.forward_kinematics(joints) @ T_EE_CAM
    positions, err = get_key_positions_in_base_frame(dets, T_bc, CAMERA_K, DIST_COEFFS)
    # positions["a"] → np.array([x, y, z]) in metres, robot base frame
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─── German ISO (QWERTZ) keyboard layout ─────────────────────────────────────
#
# Key-centre positions in the keyboard's own coordinate frame (metres).
#   Origin : centre of the ^ key (top-left of the main key block)
#   X      : rightward
#   Y      : downward
#   Z      : 0 for every key (all keys coplanar)
#
# Standard ISO pitch: 19.05 mm (3/4 inch) between adjacent key centres.
# Row y-values are exact multiples of the pitch.
# Row x-values account for the staggered widths of the left modifier keys:
#   Number row  left edge: 0       → ^ centre at 0.5 U
#   QWERTZ row  Tab 1.5 U wide     → Q centre at 2.0 U
#   ASDF   row  CapsLock 1.75 U    → A centre at 2.25 U
#   YXCVB  row  Shift 1.25 U + ISO → Y centre at 2.75 U
#
# German-specific details:
#   QWERTZ: Z at (7.0, 1.0) U — the physical "Y" column on QWERTY boards
#            Y at (2.75, 3.0) U — the physical "Z" column on QWERTY boards
#   Extra keys: ü ö ä ß ^ ´ # < (ISO-specific between shift and Y)

_P = 0.01905  # 19.05 mm in metres


def _u(col: float, row: float) -> Tuple[float, float]:
    return col * _P, row * _P


LAYOUT: Dict[str, Tuple[float, float]] = {
    # ── Function-key row (y = –1 U above the number row) ──────────────────────
    # Gaps between groups are approximated; exact values vary by keyboard model.
    "esc":  _u(0.5,  -1.0),
    "f1":   _u(2.0,  -1.0),
    "f2":   _u(3.0,  -1.0),
    "f3":   _u(4.0,  -1.0),
    "f4":   _u(5.0,  -1.0),
    "f5":   _u(6.5,  -1.0),
    "f6":   _u(7.5,  -1.0),
    "f7":   _u(8.5,  -1.0),
    "f8":   _u(9.5,  -1.0),
    "f9":   _u(11.0, -1.0),
    "f10":  _u(12.0, -1.0),
    "f11":  _u(13.0, -1.0),
    "f12":  _u(14.0, -1.0),

    # ── Number row (y = 0) ────────────────────────────────────────────────────
    "^":         _u(0.5,  0.0),
    "1":         _u(1.5,  0.0),
    "2":         _u(2.5,  0.0),
    "3":         _u(3.5,  0.0),
    "4":         _u(4.5,  0.0),
    "5":         _u(5.5,  0.0),
    "6":         _u(6.5,  0.0),
    "7":         _u(7.5,  0.0),
    "8":         _u(8.5,  0.0),
    "9":         _u(9.5,  0.0),
    "0":         _u(10.5, 0.0),
    "ß":         _u(11.5, 0.0),
    "´":         _u(12.5, 0.0),   # acute accent / backtick position
    "backspace": _u(14.0, 0.0),   # 2 U key; centre at 14.0 U

    # ── QWERTZ row (y = 1) ────────────────────────────────────────────────────
    "tab": _u(0.75, 1.0),          # 1.5 U wide
    "q":   _u(2.0,  1.0),
    "w":   _u(3.0,  1.0),
    "e":   _u(4.0,  1.0),
    "r":   _u(5.0,  1.0),
    "t":   _u(6.0,  1.0),
    "z":   _u(7.0,  1.0),          # German Z (QWERTY "Y" column)
    "u":   _u(8.0,  1.0),
    "i":   _u(9.0,  1.0),
    "o":   _u(10.0, 1.0),
    "p":   _u(11.0, 1.0),
    "ü":   _u(12.0, 1.0),
    "+":   _u(13.0, 1.0),
    # ISO Enter is L-shaped spanning rows 1 and 2.
    # Its visual centroid (used for keycap detection) sits roughly here:
    "enter": _u(14.25, 1.5),

    # ── ASDF row (y = 2) ──────────────────────────────────────────────────────
    "caps": _u(0.875, 2.0),        # CapsLock 1.75 U wide
    "a":    _u(2.25,  2.0),
    "s":    _u(3.25,  2.0),
    "d":    _u(4.25,  2.0),
    "f":    _u(5.25,  2.0),
    "g":    _u(6.25,  2.0),
    "h":    _u(7.25,  2.0),
    "j":    _u(8.25,  2.0),
    "k":    _u(9.25,  2.0),
    "l":    _u(10.25, 2.0),
    "ö":    _u(11.25, 2.0),
    "ä":    _u(12.25, 2.0),
    "#":    _u(13.25, 2.0),        # ISO-specific key right of Ä

    # ── YXCVB row (y = 3) ────────────────────────────────────────────────────
    "shift": _u(0.625, 3.0),       # Left Shift 1.25 U (also matched as "shift_l")
    "<":     _u(1.75,  3.0),       # ISO-specific key between Shift and Y
    "y":     _u(2.75,  3.0),       # German Y (QWERTY "Z" column)
    "x":     _u(3.75,  3.0),
    "c":     _u(4.75,  3.0),
    "v":     _u(5.75,  3.0),
    "b":     _u(6.75,  3.0),
    "n":     _u(7.75,  3.0),
    "m":     _u(8.75,  3.0),
    ",":     _u(9.75,  3.0),
    ".":     _u(10.75, 3.0),
    "-":     _u(11.75, 3.0),
    "shift_r": _u(13.625, 3.0),   # Right Shift 2.75 U

    # ── Space row (y = 4) ────────────────────────────────────────────────────
    "space": _u(6.875, 4.0),       # 6.25 U spacebar; others rarely detected
}

# Alternative label spellings the detector may return
_ALIASES: Dict[str, str] = {
    "capslock":   "caps",
    "cap":        "caps",
    "shift_l":    "shift",
    "left_shift": "shift",
    "lshift":     "shift",
    "right_shift":"shift_r",
    "rshift":     "shift_r",
    "bksp":       "backspace",
    "bsp":        "backspace",
    "return":     "enter",
    "acute":      "´",
    "caret":      "^",
    "tilde":      "^",
    "grave":      "^",
    "minus":      "-",
    "period":     ".",
    "comma":      ",",
    "slash":      "-",
    "plus":       "+",
    "hash":       "#",
    "less":       "<",
    "altgr":      "altgr",
    "alt_gr":     "altgr",
}


def _normalize(label: str) -> str:
    s = label.strip().lower()
    return _ALIASES.get(s, s)


# ─── Public helpers ───────────────────────────────────────────────────────────

def detections_from_roboflow(preds: list[dict]) -> Dict[str, Tuple[float, float]]:
    """
    Convert a Roboflow prediction list to {label: (u, v)} pixel centroids.

    Uses the highest-confidence detection per unique label.
    Skips the 'keyboard' whole-board bounding box class.
    """
    best: Dict[str, dict] = {}
    for p in preds:
        cls = p["class"]
        if cls.lower() == "keyboard":
            continue
        if cls not in best or p["confidence"] > best[cls]["confidence"]:
            best[cls] = p
    return {cls: (float(p["x"]), float(p["y"])) for cls, p in best.items()}


def solve_keyboard_pose(
    detections: Dict[str, Tuple[float, float]],
    camera_k: np.ndarray,
    dist_coeffs: np.ndarray,
    min_points: int = 6,
    ransac_px: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """
    Estimate keyboard pose in camera frame via PnP.

    Parameters
    ----------
    detections  : {label: (u, v)} pixel centroids from the detector
    camera_k    : 3×3 camera intrinsic matrix
    dist_coeffs : distortion coefficients
    min_points  : minimum number of layout-matched keys required
    ransac_px   : RANSAC inlier reprojection threshold (pixels)

    Returns
    -------
    rvec        : (3,1) Rodrigues rotation  — keyboard frame → camera frame
    tvec        : (3,1) translation (metres) — keyboard frame → camera frame
    mean_err    : mean reprojection error over all matched points (pixels)
    inlier_keys : list of key labels accepted as inliers by RANSAC

    Raises
    ------
    ValueError  : insufficient matches or RANSAC fails
    """
    obj_pts: List[List[float]] = []
    img_pts: List[List[float]] = []
    labels:  List[str]         = []

    for raw_label, (u, v) in detections.items():
        key = _normalize(raw_label)
        if key in LAYOUT:
            x_m, y_m = LAYOUT[key]
            obj_pts.append([x_m, y_m, 0.0])
            img_pts.append([float(u), float(v)])
            labels.append(key)

    n = len(obj_pts)
    if n < min_points:
        raise ValueError(
            f"Only {n} detected keys matched the German ISO layout "
            f"(need ≥{min_points}). Matched: {labels}"
        )

    obj = np.array(obj_pts, dtype=np.float64)
    img = np.array(img_pts, dtype=np.float64)

    # RANSAC: robust against mis-labelled or partially visible keys
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img,
        camera_k, dist_coeffs,
        confidence=0.99,
        reprojectionError=ransac_px,
        iterationsCount=2000,
    )
    if not ok or inliers is None or len(inliers) < 4:
        raise ValueError(
            f"solvePnPRansac failed (matched {n} keys, "
            f"inliers={len(inliers) if inliers is not None else 0}). "
            "Try reducing ransac_px or capturing more keys."
        )

    idx = inliers.flatten()

    # Iterative refinement on the inlier set minimises reprojection error
    _, rvec, tvec = cv2.solvePnP(
        obj[idx], img[idx],
        camera_k, dist_coeffs,
        rvec=rvec, tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    # Reprojection error across *all* matched points (not just inliers)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_k, dist_coeffs)
    mean_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))

    inlier_keys = [labels[i] for i in idx]
    print(
        f"[PnP]  matched={n}  inliers={len(idx)}  "
        f"reproj={mean_err:.2f}px  keys={inlier_keys}"
    )
    return rvec, tvec, mean_err, inlier_keys


def get_key_positions_in_base_frame(
    detections: Dict[str, Tuple[float, float]],
    T_base_cam: np.ndarray,
    camera_k: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Compute 3D positions of all keyboard keys in robot base frame.

    Parameters
    ----------
    detections  : {label: (u, v)} pixel centroids (from detections_from_roboflow)
    T_base_cam  : 4×4 rigid transform — camera frame → robot base frame
                  (compute as  FK(q) @ T_EE_CAM  from click_to_move.py)
    camera_k    : 3×3 camera intrinsic matrix
    dist_coeffs : distortion coefficients

    Returns
    -------
    positions   : {label: np.array([x, y, z])} metres in robot base frame,
                  for *every* key in the LAYOUT dict (not just detected ones)
    reproj_err  : mean reprojection error (pixels) — quality indicator
    """
    rvec, tvec, err, _ = solve_keyboard_pose(detections, camera_k, dist_coeffs)

    # Build T_cam_keyboard (keyboard frame → camera frame, metres)
    R, _ = cv2.Rodrigues(rvec)
    T_cam_keyboard = np.eye(4, dtype=np.float64)
    T_cam_keyboard[:3, :3] = R
    T_cam_keyboard[:3, 3]  = tvec.flatten()

    T_base_keyboard = T_base_cam @ T_cam_keyboard

    positions: Dict[str, np.ndarray] = {}
    for label, (x_m, y_m) in LAYOUT.items():
        p_kboard = np.array([x_m, y_m, 0.0, 1.0])
        p_base   = T_base_keyboard @ p_kboard
        positions[label] = p_base[:3].copy()

    return positions, err


def locate_key_in_base_frame(
    key: str,
    detections: Dict[str, Tuple[float, float]],
    T_base_cam: np.ndarray,
    camera_k: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Return the 3D robot-base-frame position (metres) of a single named key.

    Raises ValueError if the key is not in the German ISO layout or PnP fails.
    """
    canonical = _normalize(key)
    if canonical not in LAYOUT:
        raise ValueError(
            f"Key '{key}' → '{canonical}' is not in the German ISO QWERTZ layout. "
            f"Available keys: {sorted(LAYOUT)}"
        )
    positions, _ = get_key_positions_in_base_frame(
        detections, T_base_cam, camera_k, dist_coeffs
    )
    return positions[canonical]


# ─── Debug visualisation ──────────────────────────────────────────────────────

def draw_pnp_overlay(
    frame: np.ndarray,
    detections: Dict[str, Tuple[float, float]],
    camera_k: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Draw PnP debug overlay on frame (in-place).

    Blue dots + labels : raw detector centroids
    Green dots         : reprojected layout positions (from PnP pose)
    Top-left text      : reprojection error and inlier count
    """
    # Raw detections
    for raw_label, (u, v) in detections.items():
        cv2.circle(frame, (int(u), int(v)), 5, (255, 80, 0), -1, cv2.LINE_AA)
        cv2.putText(frame, raw_label, (int(u) + 6, int(v) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 140, 0), 1, cv2.LINE_AA)

    try:
        rvec, tvec, err, inliers = solve_keyboard_pose(detections, camera_k, dist_coeffs)
    except ValueError as exc:
        cv2.putText(frame, f"PnP failed: {exc}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 255), 1, cv2.LINE_AA)
        return frame

    # Reproject every layout key
    obj_all = np.array([[x, y, 0.0] for x, y in LAYOUT.values()], dtype=np.float64)
    proj, _ = cv2.projectPoints(obj_all, rvec, tvec, camera_k, dist_coeffs)
    h, w = frame.shape[:2]
    for label, pt in zip(LAYOUT.keys(), proj.reshape(-1, 2)):
        u, v = int(pt[0]), int(pt[1])
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(frame, (u, v), 4, (0, 220, 60), -1, cv2.LINE_AA)
            cv2.putText(frame, label, (u + 3, v - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 180, 40), 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"PnP  reproj={err:.1f}px  inliers={len(inliers)}",
        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 60), 1, cv2.LINE_AA,
    )
    return frame


# ─── Smoke-test / live debug entry point ──────────────────────────────────────

def _smoke_test() -> None:
    print("keyboard_pnp.py  layout sanity check")
    print(f"  {len(LAYOUT)} keys defined")
    print(f"  Pitch: {_P*1000:.2f} mm")

    def dist(a: str, b: str) -> float:
        xa, ya = LAYOUT[a]; xb, yb = LAYOUT[b]
        return ((xb - xa)**2 + (yb - ya)**2)**0.5 * 1000

    print(f"  ^ to 1  distance: {dist('^','1'):.2f} mm  (expect 19.05)")
    print(f"  Q to W  distance: {dist('q','w'):.2f} mm  (expect 19.05)")
    print(f"  A to S  distance: {dist('a','s'):.2f} mm  (expect 19.05)")
    print(f"  Q to A  distance: {dist('q','a'):.2f} mm  (expect ~19.6, staggered)")


def _live_debug(
    camera: int = 1,
    width: int = 640,
    height: int = 480,
    threshold: float = 0.50,
) -> None:
    """
    Open the wrist camera, run the Roboflow detector, and display the PnP
    overlay in real time.  Does NOT require the robot to be connected.

    Blue  dots : raw detector centroids
    Green dots : reprojected layout positions from PnP fit
    Cyan  text : per-key label at reprojected position

    Press q or Esc to quit, s to print the current solved pose to stdout.
    """
    import base64
    import os
    import threading
    import time
    import uuid
    from io import BytesIO
    from pathlib import Path

    import requests
    from PIL import Image

    # ── Load API key from .env.inference ──────────────────────────────────────
    _env = Path(__file__).parent / "keyboard_detection" / ".env.inference"
    if _env.exists():
        for _ln in _env.read_text().splitlines():
            _ln = _ln.strip()
            if _ln and not _ln.startswith("#") and "=" in _ln:
                _k, _v = _ln.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

    api_key    = os.environ.get("ROBOFLOW_API_KEY", "")
    host       = os.environ.get("INFERENCE_HOST",   "http://localhost:9001")
    model_id   = os.environ.get("ROBOFLOW_MODEL_ID","keyboard-key-recognition-kw7nc/14")

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not set.\n"
            f"Add it to  {_env}"
        )

    # Camera intrinsics — imported from project constants
    from click_to_move import CAMERA_K, DIST_COEFFS

    def _encode(frame: np.ndarray) -> str:
        buf = BytesIO()
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
            buf, format="JPEG", quality=95
        )
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _detect(frame: np.ndarray) -> list[dict]:
        resp = requests.post(
            f"{host.rstrip('/')}/infer/object_detection",
            json={
                "id":         str(uuid.uuid4()),
                "model_id":   model_id,
                "api_key":    api_key,
                "image":      {"type": "base64", "value": _encode(frame)},
                "confidence": threshold,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("predictions", [])

    # ── Async detection worker ────────────────────────────────────────────────
    class _Worker(threading.Thread):
        def __init__(self) -> None:
            super().__init__(daemon=True)
            self._lock      = threading.Lock()
            self._pending   = None
            self._preds:    list[dict] = []
            self._fps       = 0.0
            self._err       = ""
            self._new_frame = threading.Event()
            self._stop      = threading.Event()

        def post(self, frame: np.ndarray) -> None:
            with self._lock:
                self._pending = frame
            self._new_frame.set()

        def get(self):
            with self._lock:
                return list(self._preds), self._fps, self._err

        def stop(self) -> None:
            self._stop.set()
            self._new_frame.set()

        def run(self) -> None:
            while not self._stop.is_set():
                if not self._new_frame.wait(timeout=0.1):
                    continue
                self._new_frame.clear()
                with self._lock:
                    frame, self._pending = self._pending, None
                if frame is None:
                    continue
                t0 = time.perf_counter()
                try:
                    preds = _detect(frame)
                    fps   = 1.0 / max(time.perf_counter() - t0, 1e-3)
                    with self._lock:
                        self._preds = preds
                        self._fps   = fps
                        self._err   = ""
                except Exception as exc:
                    with self._lock:
                        self._err = str(exc)[:80]

    # ── Draw detection boxes (blue) + PnP overlay (green) ────────────────────
    def _draw(frame: np.ndarray, preds: list[dict], det_fps: float, det_err: str) -> np.ndarray:
        h, w = frame.shape[:2]

        # Raw bounding boxes from detector
        for p in preds:
            lbl = p["class"]
            if lbl.lower() == "keyboard":
                continue
            px, py = float(p["x"]), float(p["y"])
            bw, bh = float(p["width"]), float(p["height"])
            x1, y1 = int(px - bw / 2), int(py - bh / 2)
            x2, y2 = int(px + bw / 2), int(py + bh / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 255), 1, cv2.LINE_AA)
            cv2.circle(frame,    (int(px), int(py)),  3, (255, 80,  0), -1, cv2.LINE_AA)
            cv2.putText(frame, lbl, (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 255), 1, cv2.LINE_AA)

        # PnP overlay
        if preds:
            dets = detections_from_roboflow(preds)
            draw_pnp_overlay(frame, dets, CAMERA_K, DIST_COEFFS)

        # Stats bar
        n_keys = sum(1 for p in preds if p["class"].lower() != "keyboard")
        if det_err:
            cv2.putText(frame, f"DET ERR: {det_err[:55]}", (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 80, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"det {det_fps:.1f} Hz  |  {n_keys} key(s)",
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "q=quit  s=print pose", (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 220, 180), 1, cv2.LINE_AA)
        return frame

    # ── Camera + main loop ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {camera}  {aw}x{ah}  |  model: {model_id}")
    print("Blue = detector bbox    Green = PnP reprojection")
    print("High green/blue alignment = good PnP fit\n")

    worker = _Worker()
    worker.start()

    cv2.namedWindow("keyboard_pnp  debug", cv2.WINDOW_NORMAL)
    last_preds: list[dict] = []
    last_fps = 0.0
    last_err = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, "No camera signal", (40, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 200), 2)
        else:
            worker.post(frame)

        new_preds, new_fps, new_err = worker.get()
        if new_preds:
            last_preds, last_fps, last_err = new_preds, new_fps, new_err

        cv2.imshow("keyboard_pnp  debug",
                   _draw(frame.copy(), last_preds, last_fps, last_err))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s") and last_preds:
            dets = detections_from_roboflow(last_preds)
            try:
                rvec, tvec, err, inliers = solve_keyboard_pose(dets, CAMERA_K, DIST_COEFFS)
                R, _ = cv2.Rodrigues(rvec)
                print("\n--- Keyboard pose in camera frame ---")
                print(f"  tvec (m): [{tvec[0,0]:+.4f}, {tvec[1,0]:+.4f}, {tvec[2,0]:+.4f}]")
                print(f"  reproj  : {err:.2f} px   inliers: {inliers}")
                print(f"  R:\n{R}")
            except ValueError as exc:
                print(f"  PnP error: {exc}")

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    _smoke_test()

    if "--live" in sys.argv:
        # Parse optional overrides:  --live [--camera N] [--threshold F]
        _cam   = int(sys.argv[sys.argv.index("--camera")    + 1]) if "--camera"    in sys.argv else 1
        _thr   = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 0.50
        _live_debug(camera=_cam, threshold=_thr)
    elif "--list" in sys.argv:
        for k, (x, y) in sorted(LAYOUT.items()):
            print(f"  {k:12s}  ({x*1000:6.2f}, {y*1000:6.2f}) mm")
