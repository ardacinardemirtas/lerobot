#!/usr/bin/env python3
"""
keyboard_press_gui_pnp.py — Live keyboard detection GUI with PnP-based pressing.

Drop-in replacement for keyboard_press_gui.py.  The only behavioural difference
is that key positions are derived from solvePnP over the full visible key set
instead of a single-key ray–plane intersection at a fixed Z.

Controls
--------
    Left-click on a key box    two-step PnP press
    Right-click                cancel in-progress move
    a-z / 0-9                  press that key on the robot keyboard
    space / enter / backspace / tab   press those keys on the robot keyboard
    HOME key                   return to reset home position
    F5                         find keyboard home (iterative PnP)
    ` (backtick)               toggle PnP overlay (green reprojected keys)
    Esc                        quit
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
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    PORT,
    ROBOT_ID,
    CALIBRATION_DIR,
    URDF_PATH,
    CAMERA_INDEX,
    HOME_DEG,
    CAMERA_K,
    DIST_COEFFS,
)
from move_to_position_qp import (  # noqa: E402
    SO101Kinematics,
    smooth_move,
)
from press_key_pnp import (  # noqa: E402
    detect_keys,
    press_key,
    find_kb_home,
    ROBOFLOW_API_KEY,
    INTERMEDIATE_OFFSET_M,
    HOVER_OFFSET_M,
    KB_HOME_HEIGHT_M,
)

# Physical QWERTY key → QWERTZ label sent to the robot.
# On a German (QWERTZ) target keyboard, Y and Z are swapped vs QWERTY.
_QWERTY_TO_QWERTZ: dict[str, str] = {
    "y": "z",
    "z": "y",
}
from keyboard_pnp import (  # noqa: E402
    detections_from_roboflow,
    draw_pnp_overlay,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

_WIN      = "SO-101  Keyboard Press  [PnP]"
_KEY_HOME = 2359296   # VK_HOME  (0x24 << 16) on Windows
_KEY_F5   = 7667712   # VK_F5   (0x74 << 16) on Windows


# ─── Async detection worker ───────────────────────────────────────────────────

class _DetectionWorker(threading.Thread):
    """Daemon thread: runs key detection on the most recent frame posted to it."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._lock        = threading.Lock()
        self._pending:    Optional[np.ndarray] = None
        self._preds:      list[dict] = []
        self._det_fps     = 0.0
        self._det_err     = ""
        self._new_frame   = threading.Event()
        self._stop        = threading.Event()

    def post_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame
        self._new_frame.set()

    def get_state(self) -> tuple[list[dict], float, str]:
        with self._lock:
            return list(self._preds), self._det_fps, self._det_err

    def stop(self) -> None:
        self._stop.set()
        self._new_frame.set()

    def run(self) -> None:
        while not self._stop.is_set():
            if not self._new_frame.wait(timeout=0.1):
                continue
            self._new_frame.clear()
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                continue
            t0 = time.perf_counter()
            try:
                preds = detect_keys(frame)
                fps   = 1.0 / max(time.perf_counter() - t0, 1e-3)
                with self._lock:
                    self._preds   = preds
                    self._det_fps = fps
                    self._det_err = ""
            except Exception as exc:
                with self._lock:
                    self._det_err = str(exc)[:80]


# ─── GUI ─────────────────────────────────────────────────────────────────────

class KeyboardPressGUI:

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

        self._frame_lock  = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ts    = 0.0

        self._press_lock   = threading.Lock()
        self._press_thread: Optional[threading.Thread] = None
        self._cancel_evt   = threading.Event()
        self._active_key:  Optional[str] = None
        self._status       = "Ready — click a detected key to press it  [PnP mode]"

        self._show_pnp = True   # toggle with 'o'
        self._btn_home: Optional[tuple[int, int, int, int]] = None

        self._det = _DetectionWorker()
        self._det.start()

    # ── Frame provider ────────────────────────────────────────────────────────

    def _get_frame(self, fresh: bool = False) -> Optional[np.ndarray]:
        if not fresh:
            with self._frame_lock:
                return self._latest_frame.copy() if self._latest_frame is not None else None
        deadline = time.monotonic()
        while True:
            with self._frame_lock:
                if self._frame_ts > deadline and self._latest_frame is not None:
                    return self._latest_frame.copy()
            time.sleep(0.005)

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._btn_home is not None:
                x1, y1, x2, y2 = self._btn_home
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._go_home()
                    return
            label = self._hit_test(x, y)
            if label:
                self._start_press(label)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._cancel()
            self._status = "Move cancelled"

    def _hit_test(self, x: int, y: int) -> Optional[str]:
        preds, _, _ = self._det.get_state()
        for p in preds:
            if p["class"].lower() == "keyboard":
                continue
            if (p["x"] - p["width"]  / 2 <= x <= p["x"] + p["width"]  / 2 and
                    p["y"] - p["height"] / 2 <= y <= p["y"] + p["height"] / 2):
                return p["class"]
        return None

    # ── Press / cancel / home ─────────────────────────────────────────────────

    def _start_press(self, key_label: str) -> None:
        with self._press_lock:
            if self._press_thread is not None and self._press_thread.is_alive():
                self._status = "Already moving — right-click to cancel first"
                return
        self._cancel()
        with self._press_lock:
            self._active_key = key_label
            self._cancel_evt.clear()
            t = threading.Thread(
                target=self._press_worker, args=(key_label,), daemon=True
            )
            self._press_thread = t
        t.start()

    def _press_worker(self, key_label: str) -> None:
        self._status = f"[PnP] Locating '{key_label}' …"
        try:
            press_key(
                key_label,
                self.robot,
                self.kin,
                lambda: self._get_frame(fresh=True),
                lift=True,
                cancel_event=self._cancel_evt,
            )
            self._status = f"Pressed '{key_label}'"
        except Exception as exc:
            self._status = f"Error: {exc}"
        finally:
            with self._press_lock:
                self._active_key = None

    def _cancel(self) -> None:
        self._cancel_evt.set()
        with self._press_lock:
            t = self._press_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.5)
        self._cancel_evt.clear()

    def _go_home(self) -> None:
        self._cancel()
        with self._press_lock:
            self._cancel_evt.clear()
            t = threading.Thread(target=self._home_worker, daemon=True)
            self._press_thread = t
        t.start()

    def _home_worker(self) -> None:
        self._status = "Returning to reset home …"
        smooth_move(self.robot, self.kin, self._home)
        self._status = "At reset home"

    def _go_kb_home(self) -> None:
        self._cancel()
        with self._press_lock:
            self._cancel_evt.clear()
            t = threading.Thread(target=self._kb_home_worker, daemon=True)
            self._press_thread = t
        t.start()

    def _kb_home_worker(self) -> None:
        self._status = "Finding keyboard home (rough start → iterative refine) …"
        try:
            pos = find_kb_home(self.robot, self.kin,
                               lambda: self._get_frame(fresh=True))
            self._status = (f"At keyboard home  "
                            f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m  "
                            f"[{KB_HOME_HEIGHT_M*100:.0f} cm above, looking down]")
        except Exception as exc:
            self._status = f"KB_HOME failed: {exc}"

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        preds, det_fps, det_err = self._det.get_state()

        with self._press_lock:
            moving     = self._press_thread is not None and self._press_thread.is_alive()
            active_key = self._active_key

        h, w = frame.shape[:2]

        # ── PnP overlay (green reprojected layout) ────────────────────────────
        if self._show_pnp and preds:
            dets = detections_from_roboflow(preds)
            if dets:
                draw_pnp_overlay(frame, dets, CAMERA_K, DIST_COEFFS)

        # ── Detection boxes ───────────────────────────────────────────────────
        for p in preds:
            lbl  = p["class"]
            conf = float(p["confidence"])
            px   = float(p["x"]);  py = float(p["y"])
            bw   = float(p["width"]); bh = float(p["height"])
            x1 = int(px - bw / 2);  x2 = int(px + bw / 2)
            y1 = int(py - bh / 2);  y2 = int(py + bh / 2)

            if lbl.lower() == "keyboard":
                color, thick = (0, 165, 255), 2
            elif lbl == active_key:
                color, thick = (255, 80,  0), 2
            else:
                color, thick = (50,  220, 50), 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

            if lbl.lower() != "keyboard":
                cv2.circle(frame, (int(px), int(py)), 3, color, -1, cv2.LINE_AA)
                txt = f"{lbl} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                tx = max(0, min(x1, w - tw - 4))
                ty = max(th + 6, y1 - 3)
                cv2.rectangle(frame, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 3), color, -1)
                cv2.putText(frame, txt, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

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

        # ── Stats (top-left) ─────────────────────────────────────────────────
        n_keys = sum(1 for p in preds if p["class"].lower() != "keyboard")
        pnp_flag = "PnP ON" if self._show_pnp else "PnP off"
        if det_err:
            cv2.putText(frame, f"DET ERR: {det_err[:55]}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 80, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(
                frame,
                f"det {det_fps:.1f} Hz  |  {n_keys} key(s)  |  {pnp_flag}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 200), 1, cv2.LINE_AA,
            )

        # ── Status bar ───────────────────────────────────────────────────────
        bar_h = 50
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, self._status, (8, h - bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"a-z/0-9=press  right-click=cancel  ,=home  .=KB_home  `=overlay  Esc=quit  "
            f"observe={INTERMEDIATE_OFFSET_M*100:.0f}cm → hover={HOVER_OFFSET_M*100:.0f}cm",
            (8, h - bar_h + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 220, 160), 1, cv2.LINE_AA,
        )

        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WIN, self._on_mouse)

        print(f"{'─' * 55}")
        print("  SO-101 Keyboard Press GUI  [PnP mode]")
        print(f"  Camera {CAMERA_INDEX}  {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"  [F5]  Go to keyboard home ({KB_HOME_HEIGHT_M*100:.0f} cm above keyboard centre)")
        print( "  [HOME] Return to reset home")
        print( "  Click a detected key OR press a-z/0-9 to press it on the robot.")
        print( "  [` ]  Toggle PnP overlay  |  [Esc] Quit")
        print(f"{'─' * 55}\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "No camera signal", (60, CAMERA_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 200), 2)

            ts = time.monotonic()
            with self._frame_lock:
                self._latest_frame = frame.copy()
                self._frame_ts     = ts

            self._det.post_frame(frame)
            cv2.imshow(_WIN, self._draw(frame.copy()))

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            ch = key & 0xFF
            # ── Non-letter shortcuts (never conflict with robot key presses) ──
            if ch == 27:                          # Esc = quit
                break
            elif ch == ord(","):                  # , = reset home
                self._go_home()
            elif ch == ord("."):                  # . = keyboard home
                self._go_kb_home()
            elif key == _KEY_HOME:                # HOME key = reset home
                self._go_home()
            elif key == _KEY_F5:                  # F5 = keyboard home
                self._go_kb_home()
            elif ch == 96:                        # ` (backtick) = overlay
                self._show_pnp = not self._show_pnp
                self._status = f"PnP overlay {'ON' if self._show_pnp else 'OFF'}"
            # ── Physical keyboard → robot key press (all a-z and 0-9) ────────
            elif ord("a") <= ch <= ord("z"):
                label = _QWERTY_TO_QWERTZ.get(chr(ch), chr(ch))
                self._start_press(label)
            elif ord("0") <= ch <= ord("9"):
                self._start_press(chr(ch))
            elif ch == 32:                        # space
                self._start_press("space")
            elif ch == 13:                        # enter
                self._start_press("enter")
            elif ch == 8:                         # backspace
                self._start_press("backspace")
            elif ch == 9:                         # tab
                self._start_press("tab")

        self._cancel()
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

    app = KeyboardPressGUI(robot, kin, cap, home_pos=home)
    try:
        app.run()
    finally:
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
