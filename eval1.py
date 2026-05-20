#!/usr/bin/env python3
"""
eval1.py — Autonomous sequential key-press evaluation for SO-101.

Sequence : space → enter → r → l
Scoring  : 12.5 points per key pressed correctly in-order  (max 50)
Time     : 40 seconds from the moment the run is started

Controls
--------
    F5        find keyboard home (required before starting)
    s         START the timed run
    HOME      return to reset home
    ` (backtick)   toggle PnP overlay
    Esc       quit
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
from keyboard_pnp import (  # noqa: E402
    detections_from_roboflow,
    draw_pnp_overlay,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ── Eval definition ───────────────────────────────────────────────────────────
SEQUENCE      = ["space", "enter", "r", "l"]
POINTS_PER_KEY = 12.5
TIME_LIMIT_S   = 40.0

_WIN      = "SO-101  Eval-1  [space → enter → r → l]"
_KEY_HOME = 2359296   # VK_HOME
_KEY_F5   = 7667712   # VK_F5


# ─── Async detection worker (identical to keyboard_press_gui_pnp.py) ──────────

class _DetectionWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._lock       = threading.Lock()
        self._pending:   Optional[np.ndarray] = None
        self._preds:     list[dict] = []
        self._det_fps    = 0.0
        self._det_err    = ""
        self._new_frame  = threading.Event()
        self._stop       = threading.Event()

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


# ─── Main GUI ─────────────────────────────────────────────────────────────────

class Eval1GUI:

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

        self._show_pnp = True

        # ── Eval state ────────────────────────────────────────────────────────
        self._running      = False   # True while the timed run is in progress
        self._finished     = False
        self._start_t      = 0.0
        self._seq_idx      = 0       # next key to press in SEQUENCE
        self._score        = 0.0
        self._status       = "Ready — press [F5] to find keyboard home, then [s] to start"

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

    # ── Press machinery ───────────────────────────────────────────────────────

    def _is_pressing(self) -> bool:
        with self._press_lock:
            return self._press_thread is not None and self._press_thread.is_alive()

    def _start_press(self, key_label: str, on_done: Optional[callable] = None) -> None:
        with self._press_lock:
            if self._press_thread is not None and self._press_thread.is_alive():
                return
        self._cancel_evt.clear()
        with self._press_lock:
            self._active_key = key_label
            t = threading.Thread(
                target=self._press_worker, args=(key_label, on_done), daemon=True
            )
            self._press_thread = t
        t.start()

    def _press_worker(self, key_label: str, on_done: Optional[callable]) -> None:
        try:
            press_key(
                key_label,
                self.robot,
                self.kin,
                lambda: self._get_frame(fresh=True),
                lift=True,
                cancel_event=self._cancel_evt,
            )
            if on_done and not self._cancel_evt.is_set():
                on_done(success=True)
        except Exception as exc:
            self._status = f"Press error: {exc}"
            if on_done:
                on_done(success=False)
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

    # ── Home / KB home ────────────────────────────────────────────────────────

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
        self._status = "At reset home — press [F5] to find keyboard home, then [s] to start"

    def _go_kb_home(self) -> None:
        if self._running:
            return
        self._cancel()
        with self._press_lock:
            self._cancel_evt.clear()
            t = threading.Thread(target=self._kb_home_worker, daemon=True)
            self._press_thread = t
        t.start()

    def _kb_home_worker(self) -> None:
        self._status = "Finding keyboard home …"
        try:
            pos = find_kb_home(self.robot, self.kin, lambda: self._get_frame(fresh=True))
            self._status = (f"Keyboard home found "
                            f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m  "
                            f"— press [s] to START")
        except Exception as exc:
            self._status = f"KB_HOME failed: {exc}"

    # ── Eval run ──────────────────────────────────────────────────────────────

    def _start_run(self) -> None:
        if self._running or self._finished or self._is_pressing():
            return
        self._seq_idx  = 0
        self._score    = 0.0
        self._start_t  = time.monotonic()
        self._running  = True
        self._finished = False
        self._status   = f"RUN STARTED — pressing '{SEQUENCE[0]}' …"
        self._press_next()

    def _press_next(self) -> None:
        if self._seq_idx >= len(SEQUENCE):
            self._end_run(timed_out=False)
            return
        key = SEQUENCE[self._seq_idx]
        self._status = f"Pressing '{key}'  ({self._seq_idx+1}/{len(SEQUENCE)})  score={self._score:.1f}"
        self._start_press(key, on_done=self._on_key_done)

    def _on_key_done(self, success: bool) -> None:
        if not self._running:
            return
        elapsed = time.monotonic() - self._start_t
        if elapsed > TIME_LIMIT_S:
            self._end_run(timed_out=True)
            return
        if success:
            self._score   += POINTS_PER_KEY
            self._seq_idx += 1
            if self._seq_idx >= len(SEQUENCE):
                self._end_run(timed_out=False)
            else:
                self._press_next()
        else:
            self._status = "Press failed — run aborted"
            self._running = False

    def _end_run(self, timed_out: bool) -> None:
        self._running  = False
        self._finished = True
        elapsed = time.monotonic() - self._start_t
        if timed_out:
            self._status = (f"TIME OUT after {elapsed:.1f}s  |  "
                            f"score = {self._score:.1f} / {POINTS_PER_KEY * len(SEQUENCE):.1f}")
        else:
            self._status = (f"COMPLETE in {elapsed:.1f}s  |  "
                            f"score = {self._score:.1f} / {POINTS_PER_KEY * len(SEQUENCE):.1f}")
        print(f"\n{'═'*55}")
        print(f"  EVAL RESULT: {self._status}")
        print(f"{'═'*55}\n")

    # ── Time-limit watchdog ───────────────────────────────────────────────────

    def _check_timeout(self) -> None:
        if self._running and (time.monotonic() - self._start_t) > TIME_LIMIT_S:
            self._cancel()
            self._end_run(timed_out=True)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        preds, det_fps, det_err = self._det.get_state()

        with self._press_lock:
            active_key = self._active_key

        h, w = frame.shape[:2]

        # PnP overlay
        if self._show_pnp and preds:
            dets = detections_from_roboflow(preds)
            if dets:
                draw_pnp_overlay(frame, dets, CAMERA_K, DIST_COEFFS)

        # Detection boxes
        for p in preds:
            lbl  = p["class"]
            conf = float(p["confidence"])
            px   = float(p["x"]);  py = float(p["y"])
            bw   = float(p["width"]); bh = float(p["height"])
            x1 = int(px - bw / 2);  x2 = int(px + bw / 2)
            y1 = int(py - bh / 2);  y2 = int(py + bh / 2)

            # Highlight the next key in the sequence
            next_key = SEQUENCE[self._seq_idx] if self._seq_idx < len(SEQUENCE) else None
            if lbl.lower() == "keyboard":
                color, thick = (0, 165, 255), 2
            elif lbl == active_key:
                color, thick = (255, 80, 0), 2
            elif lbl == next_key:
                color, thick = (0, 255, 255), 2   # cyan = next target
            else:
                color, thick = (50, 220, 50), 1

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

        # ── Timer bar (top, full-width) ───────────────────────────────────────
        if self._running or self._finished:
            elapsed  = time.monotonic() - self._start_t if self._running else TIME_LIMIT_S
            fraction = min(elapsed / TIME_LIMIT_S, 1.0)
            bar_w    = int(w * fraction)
            remaining = max(TIME_LIMIT_S - elapsed, 0.0)
            bar_color = (0, 200, 255) if remaining > 10 else (0, 80, 255)
            cv2.rectangle(frame, (0, 0), (bar_w, 8), bar_color, -1)
            cv2.rectangle(frame, (0, 0), (w - 1, 8), (120, 120, 120), 1)
            time_txt = f"{remaining:.1f}s"
            cv2.putText(frame, time_txt, (w - 60, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        # ── Score + sequence display (top-left) ───────────────────────────────
        n_keys   = sum(1 for p in preds if p["class"].lower() != "keyboard")
        pnp_flag = "PnP ON" if self._show_pnp else "PnP off"
        if det_err:
            cv2.putText(frame, f"DET ERR: {det_err[:55]}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 80, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(
                frame,
                f"det {det_fps:.1f} Hz  |  {n_keys} key(s)  |  {pnp_flag}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 200), 1, cv2.LINE_AA,
            )

        # Sequence progress: draw each step with colour coding
        seq_x = 8
        for i, key in enumerate(SEQUENCE):
            if i < self._seq_idx:
                col = (0, 220, 0)    # green = done
            elif i == self._seq_idx:
                col = (0, 255, 255)  # cyan  = current
            else:
                col = (160, 160, 160)  # grey  = pending
            arrow = " → " if i < len(SEQUENCE) - 1 else ""
            label = f"[{key}]{arrow}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.putText(frame, label, (seq_x, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)
            seq_x += lw + 2

        # Score
        score_txt = f"score: {self._score:.1f} / {POINTS_PER_KEY * len(SEQUENCE):.1f}"
        cv2.putText(frame, score_txt, (8, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 80), 1, cv2.LINE_AA)

        # ── Status bar ───────────────────────────────────────────────────────
        bar_h = 50
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, self._status, (8, h - bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"[F5]=KB home  [s]=start  [HOME]=reset home  [`]=overlay  [Esc]=quit",
            (8, h - bar_h + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 220, 160), 1, cv2.LINE_AA,
        )

        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

        print(f"{'─' * 60}")
        print("  SO-101 Eval-1")
        print(f"  Sequence : {' → '.join(SEQUENCE)}")
        print(f"  Scoring  : {POINTS_PER_KEY} pts/key  (max {POINTS_PER_KEY * len(SEQUENCE):.0f})")
        print(f"  Time     : {TIME_LIMIT_S:.0f} s")
        print(f"  Camera {CAMERA_INDEX}  {CAMERA_WIDTH}×{CAMERA_HEIGHT}")
        print( "  [F5]  Find keyboard home first")
        print( "  [s]   Start timed run")
        print( "  [` ]  Toggle PnP overlay  |  [Esc] Quit")
        print(f"{'─' * 60}\n")

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
            self._check_timeout()

            cv2.imshow(_WIN, self._draw(frame.copy()))

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            ch = key & 0xFF

            if ch == 27:                    # Esc
                break
            elif ch == ord("s") and not self._running and not self._finished:
                self._start_run()
            elif key == _KEY_F5:            # F5 = KB home
                self._go_kb_home()
            elif ch == ord("."):            # . = KB home (alt)
                self._go_kb_home()
            elif key == _KEY_HOME:          # HOME = reset home
                self._go_home()
            elif ch == ord(","):            # , = reset home (alt)
                self._go_home()
            elif ch == 96:                  # ` = PnP overlay
                self._show_pnp = not self._show_pnp

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

    app = Eval1GUI(robot, kin, cap, home_pos=home)
    try:
        app.run()
    finally:
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
