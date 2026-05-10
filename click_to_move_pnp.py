#!/usr/bin/env python3
"""
click_to_move_pnp.py — Click-to-move with PnP-based key localisation.

Extends click_to_move.py:
  • Clicking on a detected key box  → exact PnP 3D position (no Z assumption)
  • Clicking elsewhere              → ray–plane at PnP-derived keyboard Z
                                       (falls back to TARGET_Z_M if no PnP)
  • Live PnP overlay on the feed    → toggle with 'o'

All jog keys, home, PID, and grid controls are identical to click_to_move.py.

Controls
--------
    Left-click on key box     move to PnP key position
    Left-click elsewhere      move to cursor position at keyboard Z
    Right-click               cancel current move
    Arrow Up/Down             jog ±X by 0.02 m
    Arrow Left/Right          jog ∓Y by 0.02 m
    Space                     jog +Z by 0.02 m
    r / HOME                  return to home position
    p                         toggle PID in settle loop
    o                         toggle PnP overlay
    s                         print EE position
    g                         toggle reprojection grid
    h                         toggle help
    q / Esc                   return home and quit
"""

import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Load .env.inference ───────────────────────────────────────────────────────
_ENV_FILE = Path(__file__).parent / "keyboard_detection" / ".env.inference"
if _ENV_FILE.exists():
    for _ln in _ENV_FILE.read_text().splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from click_to_move import (  # noqa: E402
    SO101Kinematics,
    smooth_move,
    pixel_to_robot_target,
    reproject_ground_grid,
    _joints_from_obs,
    _clamp_target,
    CAMERA_K,
    DIST_COEFFS,
    T_EE_CAM,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    PORT,
    ROBOT_ID,
    CALIBRATION_DIR,
    URDF_PATH,
    CAMERA_INDEX,
    HOME_DEG,
    TARGET_Z_M,
    WS_MIN,
    WS_MAX,
    JOG_M,
    ROLL_FIXED_DEG,
    _KEY_UP, _KEY_DOWN, _KEY_LEFT, _KEY_RIGHT, _KEY_HOME, _KEY_SPACE,
)
from press_key_pnp import detect_keys, ROBOFLOW_API_KEY  # noqa: E402
from keyboard_pnp import (  # noqa: E402
    detections_from_roboflow,
    get_key_positions_in_base_frame,
    draw_pnp_overlay,
    _normalize as _pnp_normalize,
    LAYOUT as PNP_LAYOUT,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

_WIN = "SO-101  Click-to-Move  [PnP]"

_HELP = [
    "Left-click key box        move to PnP key position",
    "Left-click elsewhere      move to cursor at keyboard Z (PnP)",
    "Right-click               cancel move",
    "Arrow Up/Down             jog +/- X  0.02 m",
    "Arrow Left/Right          jog -/+ Y  0.02 m",
    "Space                     jog +Z  0.02 m",
    "r / HOME                  return to home",
    "p                         toggle PID settle",
    "o                         toggle PnP overlay",
    "s                         print EE position",
    "g                         toggle reprojection grid",
    "h                         toggle help",
    "q / Esc                   return home and quit",
]


# ─── Async detection worker ───────────────────────────────────────────────────

class _DetectionWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._lock      = threading.Lock()
        self._pending:  Optional[np.ndarray] = None
        self._preds:    list[dict] = []
        self._fps       = 0.0
        self._err       = ""
        self._new_frame = threading.Event()
        self._stop      = threading.Event()

    def post_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame
        self._new_frame.set()

    def get_state(self) -> tuple[list[dict], float, str]:
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
                preds = detect_keys(frame)
                fps   = 1.0 / max(time.perf_counter() - t0, 1e-3)
                with self._lock:
                    self._preds = preds
                    self._fps   = fps
                    self._err   = ""
            except Exception as exc:
                with self._lock:
                    self._err = str(exc)[:80]


# ─── App ─────────────────────────────────────────────────────────────────────

class ClickToMovePnP:

    def __init__(
        self,
        robot: SO101Follower,
        kin: SO101Kinematics,
        cap: cv2.VideoCapture,
        home_pos: np.ndarray,
    ) -> None:
        self.robot = robot
        self.kin   = kin
        self.cap   = cap
        self._home = home_pos.copy()

        self._pid      = False
        self._help     = False
        self._grid     = False
        self._show_pnp = True

        self._status    = "Ready — click key=PnP move  |  click elsewhere=ray+keyboard Z  |  [h] help"
        self._click_uv: Optional[tuple[int, int]] = None
        self._target_3d: Optional[np.ndarray]     = None
        self._T_base_cam_last: Optional[np.ndarray] = None

        # Last PnP result: all key positions in base frame + surface Z
        self._pnp_positions: Optional[dict[str, np.ndarray]] = None
        self._pnp_keyboard_z: Optional[float] = None
        self._pnp_reproj: Optional[float] = None

        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock   = threading.Lock()

        self._btn_home: Optional[tuple[int, int, int, int]] = None

        self._det = _DetectionWorker()
        self._det.start()

    # ── Transforms ───────────────────────────────────────────────────────────

    def _T_base_cam(self) -> np.ndarray:
        obs = self.robot.get_observation()
        q   = _joints_from_obs(obs)
        T   = self.kin.forward_kinematics(q) @ T_EE_CAM
        self._T_base_cam_last = T
        return T

    # ── PnP helper ────────────────────────────────────────────────────────────

    def _update_pnp(self, preds: list[dict]) -> bool:
        """
        Re-run PnP with the current arm pose.
        Caches results in self._pnp_positions / _pnp_keyboard_z / _pnp_reproj.
        Returns True on success.
        """
        dets = detections_from_roboflow(preds)
        if not dets:
            return False
        try:
            T_bc = self._T_base_cam()
            positions, err = get_key_positions_in_base_frame(
                dets, T_bc, CAMERA_K, DIST_COEFFS
            )
            self._pnp_positions  = positions
            self._pnp_keyboard_z = float(np.median([p[2] for p in positions.values()]))
            self._pnp_reproj     = err
            return True
        except ValueError:
            return False

    def _key_at_pixel(self, x: int, y: int, preds: list[dict]) -> Optional[str]:
        """Return normalized key label if (x, y) falls inside a detected key box."""
        for p in preds:
            lbl = p["class"]
            if lbl.lower() == "keyboard":
                continue
            if (p["x"] - p["width"]  / 2 <= x <= p["x"] + p["width"]  / 2 and
                    p["y"] - p["height"] / 2 <= y <= p["y"] + p["height"] / 2):
                return _pnp_normalize(lbl)
        return None

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._btn_home is not None:
                x1, y1, x2, y2 = self._btn_home
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._go_home()
                    return
            self._handle_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._stop.set()
            self._status = "Move cancelled"

    def _handle_click(self, u: int, v: int) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._status = "Already moving — right-click to cancel"
                return

        preds, _, _ = self._det.get_state()

        # ── Try PnP path first ────────────────────────────────────────────────
        key_label = self._key_at_pixel(u, v, preds) if preds else None

        if key_label is not None and self._pnp_positions is not None:
            if key_label in self._pnp_positions:
                target = self._pnp_positions[key_label].copy()
                method = f"PnP key '{key_label}'"
            else:
                # Key detected but not in layout — fall through to ray-plane
                key_label = None

        if key_label is None:
            # ── Ray-plane fallback — use PnP keyboard Z if available ──────────
            z = self._pnp_keyboard_z if self._pnp_keyboard_z is not None else TARGET_Z_M
            try:
                T_bc = self._T_base_cam()
            except Exception as exc:
                self._status = f"FK error: {exc}"
                return
            target = pixel_to_robot_target(u, v, T_bc, CAMERA_K, DIST_COEFFS, z)
            if target is None:
                self._status = "Ray misses surface — check calibration"
                return
            method = f"ray-plane  z={z:+.4f} m"

        target = np.clip(target, WS_MIN, WS_MAX)
        self._click_uv  = (u, v)
        self._target_3d = target.copy()

        self._stop.clear()
        tgt_str = f"({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m"
        self._status = f"{method} → {tgt_str}"
        self._thread = threading.Thread(
            target=self._move_worker, args=(target, tgt_str), daemon=True
        )
        self._thread.start()

    # ── Jog / home ────────────────────────────────────────────────────────────

    def _go_home(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._move_worker, args=(self._home, "home"), daemon=True
        )
        self._thread.start()

    def _jog(self, dx: float = 0, dy: float = 0, dz: float = 0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        obs = self.robot.get_observation()
        q   = _joints_from_obs(obs)
        T   = self.kin.forward_kinematics(q)
        target = T[:3, 3].copy()
        target += np.array([dx, dy, dz])
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._move_worker, args=(target, "jog"), daemon=True
        )
        self._thread.start()

    def _move_worker(self, target: np.ndarray, label: str) -> None:
        tgt = f"({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:.3f}) m"
        self._status = f"{label} → {tgt} …"
        result = smooth_move(self.robot, self.kin, target, self._stop, self._pid)
        self._status = {
            "done":      f"Arrived  {tgt}",
            "cancelled": "Move cancelled",
            "max_iter":  f"Max iters — near {tgt}",
        }.get(result, result)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, preds: list[dict], det_fps: float, det_err: str) -> np.ndarray:
        h, w = frame.shape[:2]
        moving = self._thread is not None and self._thread.is_alive()

        # ── PnP overlay ───────────────────────────────────────────────────────
        if self._show_pnp and preds:
            dets = detections_from_roboflow(preds)
            if dets:
                draw_pnp_overlay(frame, dets, CAMERA_K, DIST_COEFFS)

        # ── Reprojection grid ─────────────────────────────────────────────────
        if self._grid and self._T_base_cam_last is not None:
            z = self._pnp_keyboard_z if self._pnp_keyboard_z is not None else TARGET_Z_M
            try:
                reproject_ground_grid(self._T_base_cam_last, CAMERA_K, z, frame)
            except Exception:
                pass

        # ── Detection boxes ───────────────────────────────────────────────────
        for p in preds:
            lbl = p["class"]
            if lbl.lower() == "keyboard":
                cv2.rectangle(
                    frame,
                    (int(p["x"] - p["width"] / 2), int(p["y"] - p["height"] / 2)),
                    (int(p["x"] + p["width"] / 2), int(p["y"] + p["height"] / 2)),
                    (0, 165, 255), 2, cv2.LINE_AA,
                )
                continue
            px, py = float(p["x"]), float(p["y"])
            bw, bh = float(p["width"]), float(p["height"])
            x1, y1 = int(px - bw / 2), int(py - bh / 2)
            x2, y2 = int(px + bw / 2), int(py + bh / 2)
            col = (50, 220, 50)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
            cv2.circle(frame, (int(px), int(py)), 3, col, -1, cv2.LINE_AA)

        # ── Click crosshair ───────────────────────────────────────────────────
        if self._click_uv is not None:
            u, v   = self._click_uv
            colour = (0, 150, 255) if moving else (50, 220, 50)
            cv2.circle(frame, (u, v), 11, colour, 2, cv2.LINE_AA)
            cv2.line(frame, (u - 16, v), (u + 16, v), colour, 1, cv2.LINE_AA)
            cv2.line(frame, (u, v - 16), (u, v + 16), colour, 1, cv2.LINE_AA)
            if self._target_3d is not None:
                t = self._target_3d
                cv2.putText(
                    frame,
                    f"({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m",
                    (u + 14, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA,
                )

        # ── HOME button ───────────────────────────────────────────────────────
        bw_ = 82
        x1b = w - bw_ - 6;  x2b = w - 6
        y1b = 6;              y2b = y1b + 28
        self._btn_home = (x1b, y1b, x2b, y2b)
        bcol = (50, 130, 220) if not moving else (30, 80, 140)
        cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), bcol, -1)
        cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (180, 210, 255), 1)
        cv2.putText(frame, "HOME [r]", (x1b + 6, y2b - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Stats bar (top-left) ──────────────────────────────────────────────
        n_keys = sum(1 for p in preds if p["class"].lower() != "keyboard")
        pnp_str = (f"PnP z={self._pnp_keyboard_z:+.3f}m  err={self._pnp_reproj:.1f}px"
                   if self._pnp_keyboard_z is not None else "PnP: waiting …")
        if det_err:
            cv2.putText(frame, f"DET ERR: {det_err[:55]}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 80, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(
                frame,
                f"det {det_fps:.1f} Hz  {n_keys} keys  |  {pnp_str}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 255, 200), 1, cv2.LINE_AA,
            )

        # ── Status bar (bottom) ───────────────────────────────────────────────
        bar_h = 56
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, self._status, (8, h - bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        pnp_flag  = "PnP:ON" if self._show_pnp else "PnP:off"
        pid_flag  = "PID:ON" if self._pid      else "PID:off"
        grid_flag = "GRID"   if self._grid     else ""
        hints = (f"{pid_flag}  {pnp_flag}  {grid_flag}  "
                 "arrows=jog  space=up  r=home  p=pid  o=pnp  g=grid  h=help  q=quit")
        cv2.putText(frame, hints, (8, h - bar_h + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 220, 160), 1, cv2.LINE_AA)

        # ── Help overlay ──────────────────────────────────────────────────────
        if self._help:
            for i, line in enumerate(_HELP):
                cv2.putText(frame, line, (10, 28 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 200), 1, cv2.LINE_AA)

        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WIN, self._on_mouse)

        print(f"{'─'*55}")
        print("  SO-101 Click-to-Move  [PnP mode]")
        print(f"  Camera {CAMERA_INDEX}  {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"  Wrist roll fixed at {ROLL_FIXED_DEG} deg")
        print("  Click a key box  → PnP position")
        print("  Click elsewhere  → ray-plane at keyboard Z")
        print("  Press [h] for full help.")
        print(f"{'─'*55}\n")

        last_pnp_update = 0.0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "No camera signal", (80, CAMERA_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 200), 2)

            # Feed detection worker
            self._det.post_frame(frame)
            preds, det_fps, det_err = self._det.get_state()

            # Re-run PnP every time we get fresh detections (~5–7 Hz)
            # but not more often than the detector fires
            if preds and det_fps > 0:
                now = time.monotonic()
                if now - last_pnp_update > 1.0 / max(det_fps, 1.0):
                    self._update_pnp(preds)
                    last_pnp_update = now

            cv2.imshow(_WIN, self._draw(frame.copy(), preds, det_fps, det_err))

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue

            if key == _KEY_UP:
                self._jog(dx=+JOG_M)
            elif key == _KEY_DOWN:
                self._jog(dx=-JOG_M)
            elif key == _KEY_LEFT:
                self._jog(dy=-JOG_M)
            elif key == _KEY_RIGHT:
                self._jog(dy=+JOG_M)
            elif key == _KEY_SPACE:
                self._jog(dz=+JOG_M)
            elif key == _KEY_HOME:
                self._go_home()
            elif key & 0xFF in (ord("q"), 27):
                break
            elif key & 0xFF == ord("r"):
                self._go_home()
            elif key & 0xFF == ord("p"):
                self._pid = not self._pid
                self._status = f"PID settle {'ON' if self._pid else 'OFF'}"
            elif key & 0xFF == ord("o"):
                self._show_pnp = not self._show_pnp
                self._status = f"PnP overlay {'ON' if self._show_pnp else 'OFF'}"
            elif key & 0xFF == ord("g"):
                self._grid = not self._grid
                if self._grid and self._T_base_cam_last is None:
                    try:
                        self._T_base_cam_last = self._T_base_cam()
                    except Exception:
                        pass
                z = self._pnp_keyboard_z if self._pnp_keyboard_z is not None else TARGET_Z_M
                self._status = f"Grid {'ON' if self._grid else 'OFF'}  z={z:+.4f} m"
            elif key & 0xFF == ord("h"):
                self._help = not self._help
            elif key & 0xFF == ord("s"):
                obs = self.robot.get_observation()
                q   = _joints_from_obs(obs)
                T   = self.kin.forward_kinematics(q)
                p   = T[:3, 3]
                kz  = self._pnp_keyboard_z
                print(f"EE  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}"
                      + (f"   keyboard_z={kz:+.4f}" if kz is not None else ""))
                self._T_base_cam_last = T @ T_EE_CAM

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._det.stop()
        cv2.destroyAllWindows()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    if not ROBOFLOW_API_KEY:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not set.\n"
            f"Add it to  {_ENV_FILE}"
        )

    kin = SO101Kinematics(URDF_PATH)

    robot = SO101Follower(SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        calibration_dir=CALIBRATION_DIR,
        use_degrees=True,
    ))
    robot.connect()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    else:
        print(f"[warn] Could not open camera {CAMERA_INDEX}")

    home = kin.forward_kinematics(HOME_DEG)[:3, 3].copy()
    print(f"Home: x={home[0]:+.4f}  y={home[1]:+.4f}  z={home[2]:+.4f}")

    stop = threading.Event()
    print("Moving to home …")
    smooth_move(robot, kin, home, stop, pid_enabled=False)
    print("At home.\n")

    app = ClickToMovePnP(robot, kin, cap, home_pos=home)
    try:
        app.run()
    finally:
        print("\nReturning to home …")
        stop = threading.Event()
        smooth_move(robot, kin, home, stop, pid_enabled=False)
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
