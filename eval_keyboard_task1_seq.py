#!/usr/bin/env python3
"""
eval_keyboard_task1_seq.py — Eval 1 Task (2): press Space → Enter → R → L
in sequence within 40 seconds.  12.5 points per correctly pressed key.

Usage
-----
    python eval_keyboard_task1_seq.py

Controls
--------
    F5      Find keyboard home (must do before starting)
    Space   Start episode (after KB_HOME is set)
    Esc     Quit
    ,       Return to reset home
    .       Re-find keyboard home
    `       Toggle PnP overlay
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
    WS_MIN,
    WS_MAX,
)
from move_to_position_qp_hold_orient import (  # noqa: E402
    SO101Kinematics,
    smooth_move,
)
from press_key_pnp import (  # noqa: E402
    detect_keys,
    press_key,
    find_kb_home,
    return_to_kb_home,
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

# ── Task definition ───────────────────────────────────────────────────────────
TASK_SEQUENCE = ["space", "enter", "r", "l"]
POINTS_PER_KEY = 12.5
TIME_LIMIT_S   = 40.0

_WIN      = "SO-101  Eval 1 Task (2)  [Space→Enter→R→L]"
_KEY_HOME = 2359296
_KEY_F5   = 7667712


# ── Detection worker ──────────────────────────────────────────────────────────

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


# ── Eval GUI ──────────────────────────────────────────────────────────────────

class SequenceEvalGUI:

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

        self._frame_lock   = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ts     = 0.0

        self._press_lock   = threading.Lock()
        self._press_thread: Optional[threading.Thread] = None
        self._cancel_evt   = threading.Event()
        self._active_key:  Optional[str] = None

        # Episode state
        self._kb_home_set    = False
        self._episode_active = False
        self._episode_done   = False
        self._seq_idx        = 0
        self._score          = 0.0
        self._ep_start_t     = 0.0
        self._final_time     = 0.0

        self._status    = "Finding keyboard home …"
        self._show_pnp  = True

        self._det = _DetectionWorker()
        self._det.start()

    # ── Frame provider ────────────────────────────────────────────────────────

    def _get_frame(self, fresh: bool = False, timeout: float = 5.0) -> Optional[np.ndarray]:
        if not fresh:
            with self._frame_lock:
                return self._latest_frame.copy() if self._latest_frame is not None else None
        deadline = time.monotonic()
        cutoff   = deadline + timeout
        while True:
            with self._frame_lock:
                if self._frame_ts > deadline and self._latest_frame is not None:
                    return self._latest_frame.copy()
            if time.monotonic() > cutoff:
                return None   # camera stalled (low-light auto-exposure slowdown)
            time.sleep(0.005)

    # ── Episode control ───────────────────────────────────────────────────────

    def _start_episode(self, inline: bool = False) -> None:
        """Start the episode.  Pass inline=True when called from within the
        press thread (e.g. from _kb_home_worker) to run the sequence in-place
        rather than spawning a new thread."""
        if not self._kb_home_set:
            self._status = "Set keyboard home first with [F5]"
            return
        if not inline:
            with self._press_lock:
                if self._press_thread is not None and self._press_thread.is_alive():
                    self._status = "Wait for current move to finish"
                    return
        self._seq_idx        = 0
        self._score          = 0.0
        self._episode_active = True
        self._episode_done   = False
        self._ep_start_t     = time.monotonic()
        self._status         = f"Episode started — pressing '{TASK_SEQUENCE[0]}' …"
        self._cancel_evt.clear()
        if inline:
            self._sequence_worker()
        else:
            t = threading.Thread(target=self._sequence_worker, daemon=True)
            with self._press_lock:
                self._press_thread = t
            t.start()

    def _sequence_worker(self) -> None:
        for idx, key in enumerate(TASK_SEQUENCE):
            # Time check before each press
            elapsed = time.monotonic() - self._ep_start_t
            if elapsed >= TIME_LIMIT_S:
                self._status = f"Time's up!  Score: {self._score:.1f} / {len(TASK_SEQUENCE)*POINTS_PER_KEY:.1f}"
                self._episode_active = False
                self._episode_done   = True
                self._final_time     = elapsed
                return

            remaining = TIME_LIMIT_S - elapsed
            self._status = (
                f"[{idx+1}/{len(TASK_SEQUENCE)}]  Pressing '{key}' …"
                f"  score: {self._score:.0f}  time left: {remaining:.0f}s"
            )
            _MAX_ATTEMPTS = 3
            pressed       = False

            for attempt in range(_MAX_ATTEMPTS):
                if self._cancel_evt.is_set():
                    break
                if time.monotonic() - self._ep_start_t >= TIME_LIMIT_S:
                    break

                with self._press_lock:
                    self._active_key = key

                try:
                    press_key(
                        key,
                        self.robot,
                        self.kin,
                        lambda: self._get_frame(fresh=True),
                        lift=False,
                        cancel_event=self._cancel_evt,
                    )
                    pressed = True
                except Exception as exc:
                    self._status = (
                        f"Press failed [{key}] "
                        f"({attempt + 1}/{_MAX_ATTEMPTS}): {str(exc)[:50]}"
                    )
                    if attempt < _MAX_ATTEMPTS - 1:
                        try:
                            return_to_kb_home(self.robot)
                        except Exception:
                            pass
                finally:
                    with self._press_lock:
                        self._active_key = None

                if pressed:
                    break

            if not pressed:
                self._episode_active = False
                self._episode_done   = True
                self._final_time     = time.monotonic() - self._ep_start_t
                return

            if self._cancel_evt.is_set():
                self._status = "Episode cancelled"
                self._episode_active = False
                self._episode_done   = True
                self._final_time     = time.monotonic() - self._ep_start_t
                return

            self._score  += POINTS_PER_KEY
            self._seq_idx = idx + 1

            # Return to KB_HOME between presses (not after the last one)
            if idx < len(TASK_SEQUENCE) - 1:
                self._status = (
                    f"Pressed '{key}' ✓  score: {self._score:.0f}"
                    f"  — returning to KB_HOME …"
                )
                return_to_kb_home(self.robot)

        elapsed = time.monotonic() - self._ep_start_t
        self._status = (
            f"Complete!  Score: {self._score:.0f} / {len(TASK_SEQUENCE)*POINTS_PER_KEY:.0f}"
            f"  in {elapsed:.1f}s"
        )
        self._episode_active = False
        self._episode_done   = True
        self._final_time     = elapsed
        print(f"\n[Eval] Score: {self._score:.0f}/{len(TASK_SEQUENCE)*POINTS_PER_KEY:.0f}  "
              f"time: {elapsed:.1f}s")

    # ── Utility moves ─────────────────────────────────────────────────────────

    def _cancel(self) -> None:
        self._cancel_evt.set()
        with self._press_lock:
            t = self._press_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._cancel_evt.clear()

    def _go_home(self) -> None:
        self._cancel()
        self._episode_active = False
        with self._press_lock:
            self._cancel_evt.clear()
            t = threading.Thread(target=self._home_worker, daemon=True)
            self._press_thread = t
        t.start()

    def _home_worker(self) -> None:
        self._status = "Returning to reset home …"
        smooth_move(self.robot, self.kin, self._home)
        self._status = "At reset home — press [F5] to set KB_HOME, then [Space] to start"

    def _go_kb_home(self) -> None:
        self._cancel()
        self._episode_active = False
        with self._press_lock:
            self._cancel_evt.clear()
            t = threading.Thread(target=self._kb_home_worker, daemon=True)
            self._press_thread = t
        t.start()

    def _kb_home_worker(self) -> None:
        self._status = "Finding keyboard home …"
        try:
            pos = find_kb_home(self.robot, self.kin,
                               lambda: self._get_frame(fresh=True))
            self._kb_home_set = True
            self._status = (
                f"KB_HOME set  ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m"
                f"  — starting episode …"
            )
        except Exception as exc:
            self._status = f"KB_HOME failed: {exc}"
            return
        # Auto-start episode immediately after KB_HOME is found (inline — we're
        # already inside the press thread, so no new thread needed)
        self._start_episode(inline=True)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        preds, det_fps, det_err = self._det.get_state()

        with self._press_lock:
            moving     = self._press_thread is not None and self._press_thread.is_alive()
            active_key = self._active_key

        h, w = frame.shape[:2]

        if self._show_pnp and preds:
            dets = detections_from_roboflow(preds)
            if dets:
                draw_pnp_overlay(frame, dets, CAMERA_K, DIST_COEFFS)

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
                color, thick = (255, 80, 0), 2
            elif lbl in TASK_SEQUENCE:
                color, thick = (80, 200, 255), 1   # highlight task keys
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

        # ── Score / sequence panel (top-left) ─────────────────────────────────
        elapsed   = time.monotonic() - self._ep_start_t if self._episode_active else self._final_time
        remaining = max(0.0, TIME_LIMIT_S - elapsed) if self._episode_active else 0.0

        panel_y = 8
        # Sequence progress
        seq_parts = []
        for i, k in enumerate(TASK_SEQUENCE):
            if i < self._seq_idx:
                seq_parts.append(f"[{k.upper()} ✓]")
            elif i == self._seq_idx and self._episode_active:
                seq_parts.append(f"[{k.upper()} ←]")
            else:
                seq_parts.append(f"[{k.upper()}]")
        seq_str = "  ".join(seq_parts)

        cv2.putText(frame, seq_str, (8, panel_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 230, 100), 1, cv2.LINE_AA)

        score_str = f"Score: {self._score:.0f} / {len(TASK_SEQUENCE)*POINTS_PER_KEY:.0f}"
        cv2.putText(frame, score_str, (8, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 100), 1, cv2.LINE_AA)

        # ── Timer (top-right) ─────────────────────────────────────────────────
        if self._episode_active:
            timer_color = (0, 80, 255) if remaining < 10.0 else (200, 255, 200)
            timer_str   = f"{remaining:.1f}s"
        elif self._episode_done:
            timer_color = (100, 255, 100)
            timer_str   = f"{self._final_time:.1f}s"
        else:
            timer_color = (160, 160, 160)
            timer_str   = f"{TIME_LIMIT_S:.0f}s"
        (tw, _), _ = cv2.getTextSize(timer_str, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
        cv2.putText(frame, timer_str, (w - tw - 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, timer_color, 2, cv2.LINE_AA)

        # ── Det stats ─────────────────────────────────────────────────────────
        n_keys = sum(1 for p in preds if p["class"].lower() != "keyboard")
        pnp_flag = "PnP ON" if self._show_pnp else "PnP off"
        if det_err:
            cv2.putText(frame, f"DET ERR: {det_err[:55]}", (8, panel_y + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (40, 80, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(
                frame,
                f"det {det_fps:.1f} Hz  |  {n_keys} key(s)  |  {pnp_flag}",
                (8, panel_y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA,
            )

        # ── Status bar ────────────────────────────────────────────────────────
        bar_h = 50
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, self._status, (8, h - bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Space=restart  ,=home  .=KB_home  `=overlay  Esc=quit  "
            f"observe={INTERMEDIATE_OFFSET_M*100:.0f}cm → hover={HOVER_OFFSET_M*100:.0f}cm",
            (8, h - bar_h + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 220, 160), 1, cv2.LINE_AA,
        )

        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

        print(f"{'─' * 60}")
        print("  Eval 1 Task (2) — Space → Enter → R → L")
        print(f"  Sequence: {' → '.join(k.upper() for k in TASK_SEQUENCE)}")
        print(f"  Points per key: {POINTS_PER_KEY}  |  Time limit: {TIME_LIMIT_S}s")
        print(f"  Camera {CAMERA_INDEX}  {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print( "  Finding keyboard home and starting automatically …")
        print(f"{'─' * 60}\n")

        # Auto-find KB_HOME and start episode without any user input
        self._go_kb_home()

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

            # Auto-timeout check
            if (self._episode_active
                    and ts - self._ep_start_t >= TIME_LIMIT_S):
                self._cancel()
                self._episode_active = False
                self._episode_done   = True
                self._final_time     = TIME_LIMIT_S
                self._status = (
                    f"Time's up!  Score: {self._score:.0f}"
                    f" / {len(TASK_SEQUENCE)*POINTS_PER_KEY:.0f}"
                )

            self._det.post_frame(frame)
            cv2.imshow(_WIN, self._draw(frame.copy()))

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            ch = key & 0xFF

            if ch == 27:                       # Esc
                break
            elif ch == ord(","):               # , = reset home
                self._go_home()
            elif ch == ord("."):               # . = keyboard home
                self._go_kb_home()
            elif key == _KEY_HOME:
                self._go_home()
            elif key == _KEY_F5:
                self._go_kb_home()
            elif ch == 96:                     # ` = overlay
                self._show_pnp = not self._show_pnp
                self._status = f"PnP overlay {'ON' if self._show_pnp else 'OFF'}"
            elif ch == 32:                     # Space = restart episode manually
                if not self._episode_active:
                    self._go_kb_home()         # re-find KB_HOME then auto-start

        self._cancel()
        self._det.stop()
        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

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

    app = SequenceEvalGUI(robot, kin, cap, home_pos=home)
    try:
        app.run()
    finally:
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
