#!/usr/bin/env python
"""
projection_verify.py — Verify the pixel→robot 3D projection pipeline.

The robot is kept in compliant (torque-off) mode throughout so you can
move it freely by hand.

Workflow:
  1. Touch the EE tip to any point on the table.
     Press  T  → records FK position as ground truth for that point.
  2. Gently move the arm so the camera looks at the same point on the table.
  3. Press  F  (or right-click) → freezes the frame and captures T_base_cam.
  4. Left-click on the same point in the frozen image.
     → shows projected (x, y) vs the last ground-truth touch (x, y) and error.
  5. Press  F  → unfreeze for the next measurement.
  6. Press  S  → save all measurements to projection_verify_log.csv.
  7. Press  Q  → quit (torque stays off; power-cycle or manually re-enable).

Tips:
  • Use a small marker (tape cross, pen dot) so clicks are repeatable.
  • For best coverage: measure points at image centre, all four edges, corners.
  • The z-plane used for projection is the FK z measured at touch time,
    so TARGET_Z_M is NOT assumed — the measurement isolates x, y error only.
"""

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ikpy.chain import Chain

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ─── CONFIG — keep in sync with click_to_move.py ─────────────────────────────

PORT            = "COM5"
ROBOT_ID        = "my_awesome_follower_arm"
CALIBRATION_DIR = Path(r"C:\Users\plata\robots")
URDF_PATH       = r"C:\Users\plata\robots\lerobot\calibration\so101_new_calib.urdf"

ARM_JOINTS  = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]

CAMERA_INDEX  = 1
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

_s   = 640 / 1440
_cx0 = 981.92182023 - 240
CAMERA_K = np.array([
    [693.35550704 * _s, 0.0,                _cx0          * _s],
    [0.0,               692.62105461 * _s,  496.33606080  * _s],
    [0.0,               0.0,                1.0               ],
], dtype=float)
DIST_COEFFS = np.array(
    [0.06046540, -0.07201475, -0.00076385, 0.00059471, -0.00305710],
    dtype=float,
)

# Ground truth — verified independently by two solvers (PARK/HORAUD/DANIILIDIS + external).
T_EE_CAM = np.array([
    [-0.998972,  0.003967,  0.045150, -0.006058],
    [-0.022214, -0.911169, -0.411433,  0.068949],
    [ 0.039507, -0.412013,  0.910321, -0.057824],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
], dtype=float)

LOG_PATH = Path(__file__).parent / "projection_verify_log.csv"

# ─────────────────────────────────────────────────────────────────────────────


# ─── Kinematics (forward only) ────────────────────────────────────────────────

class Kinematics:
    def __init__(self, urdf_path: str):
        probe = Chain.from_urdf_file(urdf_path)
        self._link_names = [lnk.name for lnk in probe.links]
        active = [j for j in ARM_JOINTS if j != "wrist_roll"]
        mask = [name in active for name in self._link_names]
        self.chain = Chain.from_urdf_file(urdf_path, active_links_mask=mask)
        self._arm_idx = [self._link_names.index(n) for n in ARM_JOINTS]

    def fk(self, joint_deg: np.ndarray) -> np.ndarray:
        q = np.zeros(len(self._link_names))
        for idx, deg in zip(self._arm_idx, joint_deg):
            q[idx] = np.deg2rad(deg)
        return self.chain.forward_kinematics(q)   # 4×4


def joints_from_obs(obs: dict) -> np.ndarray:
    return np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=float)


# ─── Projection ───────────────────────────────────────────────────────────────

def pixel_to_world(
    u: float, v: float,
    T_base_cam: np.ndarray,
    z_plane: float,
) -> Optional[np.ndarray]:
    """Back-project pixel (u,v) to the plane z=z_plane in robot base frame."""
    pt_u = cv2.undistortPoints(
        np.array([[[u, v]]], dtype=np.float32), CAMERA_K, DIST_COEFFS, P=CAMERA_K
    )
    u_u = float(pt_u[0, 0, 0])
    v_u = float(pt_u[0, 0, 1])

    fx, fy = CAMERA_K[0, 0], CAMERA_K[1, 1]
    cx, cy = CAMERA_K[0, 2], CAMERA_K[1, 2]

    d_cam  = np.array([(u_u - cx) / fx, (v_u - cy) / fy, 1.0])
    R, t   = T_base_cam[:3, :3], T_base_cam[:3, 3]
    d_base = R @ d_cam

    if abs(d_base[2]) < 1e-6:
        return None
    lam = (z_plane - t[2]) / d_base[2]
    if lam < 0:
        return None
    return t + lam * d_base


def world_to_pixel(P_base: np.ndarray, T_base_cam: np.ndarray) -> Optional[tuple[int, int]]:
    """Project a robot-base-frame point to pixel coordinates (for overlay)."""
    T_cam_base = np.linalg.inv(T_base_cam)
    P_cam = T_cam_base @ np.append(P_base, 1.0)
    if P_cam[2] <= 0:
        return None
    fx, fy = CAMERA_K[0, 0], CAMERA_K[1, 1]
    cx, cy = CAMERA_K[0, 2], CAMERA_K[1, 2]
    u = fx * P_cam[0] / P_cam[2] + cx
    v = fy * P_cam[1] / P_cam[2] + cy
    return int(round(u)), int(round(v))


# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class TouchPoint:
    label: int
    x: float
    y: float
    z: float          # measured EE z (= table surface z in FK frame)

    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Comparison:
    touch: TouchPoint
    u: int
    v: int
    x_proj: float
    y_proj: float
    z_proj: float     # equals touch.z by construction
    err_x_mm: float
    err_y_mm: float
    err_xy_mm: float  # 2D Euclidean in mm


# ─── App ──────────────────────────────────────────────────────────────────────

class App:
    _WIN = "Projection Verify — SO-101"

    def __init__(self, robot: SO101Follower, kin: Kinematics, cap: cv2.VideoCapture):
        self.robot = robot
        self.kin   = kin
        self.cap   = cap

        self._frozen      = False
        self._frozen_frame: Optional[np.ndarray] = None
        self._frozen_T:     Optional[np.ndarray] = None  # T_base_cam at freeze time

        self._touches: list[TouchPoint]    = []
        self._comps:   list[Comparison]    = []
        self._last_comp: Optional[Comparison] = None

        self._click_uv: Optional[tuple[int, int]] = None

    # ── robot helpers ─────────────────────────────────────────────────────────

    def _read_ee(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (joint_deg[6], T_base_ee 4×4)."""
        obs = self.robot.get_observation()
        q   = joints_from_obs(obs)
        T   = self.kin.fk(q)
        return q, T

    def _T_base_cam(self, T_base_ee: np.ndarray) -> np.ndarray:
        return T_base_ee @ T_EE_CAM

    # ── actions ───────────────────────────────────────────────────────────────

    def _record_touch(self) -> None:
        _, T = self._read_ee()
        p = T[:3, 3]
        tp = TouchPoint(len(self._touches) + 1, p[0], p[1], p[2])
        self._touches.append(tp)
        print(f"\n[Touch #{tp.label}]  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f} m")

    def _freeze(self) -> None:
        ret, frame = self.cap.read()
        if not ret:
            print("[warn] Could not grab frame for freeze.")
            return
        _, T_ee = self._read_ee()
        self._frozen_T     = self._T_base_cam(T_ee)
        self._frozen_frame = frame.copy()
        self._frozen       = True
        self._click_uv     = None
        self._last_comp    = None
        cam_pos = self._frozen_T[:3, 3]
        print(f"\n[Freeze]  Camera at  x={cam_pos[0]:+.4f}  y={cam_pos[1]:+.4f}  z={cam_pos[2]:+.4f}")
        if not self._touches:
            print("  (no touch points yet — press T while touching a surface to record one)")

    def _unfreeze(self) -> None:
        self._frozen = False

    def _handle_click(self, u: int, v: int) -> None:
        if not self._frozen or self._frozen_T is None:
            return
        if not self._touches:
            print("[click] No touch points recorded yet — press T first.")
            return

        self._click_uv = (u, v)
        tp = self._touches[-1]   # compare against the last-recorded touch

        P_proj = pixel_to_world(u, v, self._frozen_T, z_plane=tp.z)
        if P_proj is None:
            print(f"[click ({u},{v})] Ray misses the z={tp.z:.4f} plane.")
            return

        err_x  = (P_proj[0] - tp.x) * 1000
        err_y  = (P_proj[1] - tp.y) * 1000
        err_xy = float(np.hypot(err_x, err_y))

        comp = Comparison(
            touch=tp,
            u=u, v=v,
            x_proj=float(P_proj[0]), y_proj=float(P_proj[1]), z_proj=float(P_proj[2]),
            err_x_mm=err_x, err_y_mm=err_y, err_xy_mm=err_xy,
        )
        self._comps.append(comp)
        self._last_comp = comp

        print(f"\n[Click ({u:3d},{v:3d})]  vs Touch #{tp.label}")
        print(f"  Projected : x={P_proj[0]:+.4f}  y={P_proj[1]:+.4f}  (z-plane={tp.z:.4f})")
        print(f"  Truth FK  : x={tp.x:+.4f}  y={tp.y:+.4f}")
        print(f"  Error     : Δx={err_x:+.1f} mm  Δy={err_y:+.1f} mm  |err|={err_xy:.1f} mm")

    def _save_csv(self) -> None:
        if not self._comps:
            print("[save] Nothing to save yet.")
            return
        with open(LOG_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "touch_id", "u_click", "v_click",
                "x_proj", "y_proj", "z_proj",
                "x_truth", "y_truth", "z_truth",
                "err_x_mm", "err_y_mm", "err_xy_mm",
            ])
            for c in self._comps:
                w.writerow([
                    c.touch.label, c.u, c.v,
                    f"{c.x_proj:.6f}", f"{c.y_proj:.6f}", f"{c.z_proj:.6f}",
                    f"{c.touch.x:.6f}", f"{c.touch.y:.6f}", f"{c.touch.z:.6f}",
                    f"{c.err_x_mm:.2f}", f"{c.err_y_mm:.2f}", f"{c.err_xy_mm:.2f}",
                ])
        print(f"[save] {len(self._comps)} measurements → {LOG_PATH}")
        self._print_summary()

    def _print_summary(self) -> None:
        if not self._comps:
            return
        errs = np.array([c.err_xy_mm for c in self._comps])
        print(f"\n{'─'*50}")
        print(f"  {len(errs)} comparisons")
        print(f"  mean |err|  = {errs.mean():.1f} mm")
        print(f"  max  |err|  = {errs.max():.1f} mm")
        print(f"  min  |err|  = {errs.min():.1f} mm")
        print(f"  std         = {errs.std():.1f} mm")
        worst = self._comps[int(errs.argmax())]
        print(f"  worst click : ({worst.u}, {worst.v})  err={worst.err_xy_mm:.1f} mm  "
              f"→ touch #{worst.touch.label}")
        print(f"{'─'*50}\n")

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_live(self, frame: np.ndarray, T_base_ee: np.ndarray) -> np.ndarray:
        p = T_base_ee[:3, 3]
        T_bc = self._T_base_cam(T_base_ee)
        cam_p = T_bc[:3, 3]

        # Reproject known touch points into current camera view
        for tp in self._touches:
            px = world_to_pixel(tp.pos(), T_bc)
            if px is not None:
                cv2.circle(frame, px, 8, (0, 255, 128), 2)
                cv2.putText(frame, f"#{tp.label}", (px[0]+9, px[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

        # Overlays
        self._put_lines(frame, [
            f"EE  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}",
            f"CAM x={cam_p[0]:+.4f}  y={cam_p[1]:+.4f}  z={cam_p[2]:+.4f}",
        ], y0=8, colour=(200, 255, 200))

        n = len(self._touches)
        touch_str = (f"Touch #{self._touches[-1].label}: "
                     f"({self._touches[-1].x:+.4f}, {self._touches[-1].y:+.4f}, {self._touches[-1].z:+.4f})"
                     ) if n else "no touches yet"
        self._status_bar(frame, [
            f"LIVE  |  {n} touch(es)  |  {touch_str}",
            "T=record touch   F/right-click=freeze   S=save   Q=quit",
        ])
        return frame

    def _draw_frozen(self, frame: np.ndarray) -> np.ndarray:
        # Blue vignette border to signal frozen state
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (200, 100, 0), 5)

        # Reproject touch points into frozen camera view
        if self._frozen_T is not None:
            for tp in self._touches:
                px = world_to_pixel(tp.pos(), self._frozen_T)
                if px is not None:
                    col = (0, 200, 255) if tp == self._touches[-1] else (100, 140, 200)
                    cv2.circle(frame, px, 8, col, 2)
                    cv2.putText(frame, f"#{tp.label}", (px[0]+9, px[1]-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Click marker + comparison result
        if self._click_uv is not None:
            u, v = self._click_uv
            cv2.drawMarker(frame, (u, v), (0, 80, 255),
                           cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            if self._last_comp is not None:
                c = self._last_comp
                lines = [
                    f"Proj  ({c.x_proj:+.4f}, {c.y_proj:+.4f})",
                    f"Truth ({c.touch.x:+.4f}, {c.touch.y:+.4f})  touch #{c.touch.label}",
                    f"|err| {c.err_xy_mm:.1f} mm  "
                    f"(dx={c.err_x_mm:+.1f}  dy={c.err_y_mm:+.1f})",
                ]
                self._put_lines(frame, lines, y0=8, colour=(80, 180, 255))

        n = len(self._touches)
        tp_str = (f"comparing vs touch #{self._touches[-1].label}" if n
                  else "no touch recorded — unfreeze, press T first")
        self._status_bar(frame, [
            f"FROZEN  |  {n} touch(es)  |  {tp_str}",
            "click=project   F/right-click=unfreeze   S=save   Q=quit",
        ])
        return frame

    @staticmethod
    def _put_lines(
        frame: np.ndarray,
        lines: list[str],
        y0: int,
        colour: tuple,
        scale: float = 0.42,
    ) -> None:
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (8, y0 + 18 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, ln, (8, y0 + 18 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)

    @staticmethod
    def _status_bar(frame: np.ndarray, lines: list[str]) -> None:
        h = frame.shape[0]
        bar_h = 14 + 18 * len(lines)
        frame[h - bar_h:] = (frame[h - bar_h:].astype(np.float32) * 0.35).astype(np.uint8)
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (8, h - bar_h + 14 + 18 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    # ── mouse callback ────────────────────────────────────────────────────────

    def _on_mouse(self, event: int, x: int, y: int, *_) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._frozen:
                self._handle_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._frozen:
                self._unfreeze()
            else:
                self._freeze()

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._WIN, self._on_mouse)

        print(f"\n{'═'*55}")
        print("  Projection Verify — SO-101")
        print("  Robot is COMPLIANT — move freely by hand.")
        print(f"{'═'*55}")
        print("  T         record EE touch as ground truth")
        print("  F / R-clk freeze frame, then left-click the same point")
        print("  S         save comparisons to CSV")
        print("  Q / Esc   quit")
        print(f"{'═'*55}\n")

        while True:
            # ── build display frame ──────────────────────────────────────────
            if self._frozen and self._frozen_frame is not None:
                frame = self._frozen_frame.copy()
                self._draw_frozen(frame)
            else:
                ret, frame = self.cap.read()
                if not ret:
                    frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), np.uint8)
                try:
                    _, T_ee = self._read_ee()
                    self._draw_live(frame, T_ee)
                except Exception as exc:
                    cv2.putText(frame, f"FK error: {exc}", (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(self._WIN, frame)

            # ── key handling ─────────────────────────────────────────────────
            key = cv2.waitKeyEx(1)
            if key == -1:
                continue

            k = key & 0xFF
            if k in (ord("q"), 27):   # q / Esc
                break
            elif k == ord("t"):
                self._record_touch()
            elif k == ord("f"):
                if self._frozen:
                    self._unfreeze()
                else:
                    self._freeze()
            elif k == ord("s"):
                self._save_csv()

        cv2.destroyAllWindows()
        if self._comps:
            self._print_summary()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    kin = Kinematics(URDF_PATH)

    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        use_degrees=True,
    )
    robot = SO101Follower(config)
    robot.connect()

    # Go compliant immediately — all motors torque-off
    robot.bus.disable_torque()
    print("Torque DISABLED — robot is compliant.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {CAMERA_INDEX}: {actual_w}×{actual_h}")
    else:
        print(f"[warn] Camera {CAMERA_INDEX} not available.")

    app = App(robot, kin, cap)
    try:
        app.run()
    finally:
        cap.release()
        # Leave torque off — safer for the user to decide when to re-enable
        print("Done. Torque remains OFF. Disconnect power when done.")
        robot.disconnect()


if __name__ == "__main__":
    main()
