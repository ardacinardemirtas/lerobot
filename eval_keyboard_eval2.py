#!/usr/bin/env python3
"""
eval_keyboard_eval2.py — Eval 2: press any a-z key given as input within 10s.
16 rollouts × 3.125 pts = 50 pts max.

The sequence is a fixed random sentence (a-z + space).  Spaces are skipped
(robot presses the space bar, which counts as a correct press).

Usage
-----
    python eval_keyboard_eval2.py

Flow (fully automatic)
----------------------
    1. Script launches → finds KB_HOME → starts rollout 1.
    2. Robot receives the next character, presses it within 10 s.
    3. Returns to KB_HOME → next rollout.
    4. After 16 rollouts, prints final score.

Controls
--------
    Esc     Quit early
    ,       Return to reset home (cancels current rollout)
    .       Re-find keyboard home
    `       Toggle PnP overlay
"""

import os
import random
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

# Fixed random sentence (a-z + space), exactly 16 characters for 16 rollouts.
# Generated with seed=42 — identical across groups.
_SENTENCE_SEED = 42
_ROLLOUT_COUNT = 16
_ROLLOUT_TIME_S = 10.0
_POINTS_PER_ROLLOUT = 3.125

# QWERTY → QWERTZ (German layout keyboard)
_QWERTY_TO_QWERTZ: dict[str, str] = {"y": "z", "z": "y"}


def _generate_sequence(n: int = _ROLLOUT_COUNT, seed: int = _SENTENCE_SEED) -> list[str]:
    """Generate a fixed random sequence of n characters from a-z + space."""
    rng = random.Random(seed)
    chars = list("abcdefghijklmnoprstuvwx")  # a-z without y/z (swapped on QWERTZ)
    chars += ["y", "z", "space"]
    # Build sentence: words of 3-6 letters separated by spaces, total n chars
    seq: list[str] = []
    while len(seq) < n:
        word_len = rng.randint(3, 6)
        for _ in range(word_len):
            seq.append(rng.choice(list("abcdefghijklmnopqrstuvwxyz")))
            if len(seq) == n:
                break
        if len(seq) < n:
            seq.append("space")
    return seq[:n]


TASK_SEQUENCE = _generate_sequence()

_WIN      = "SO-101  Eval 2  [a-z key press]"
_KEY_HOME = 2359296
_KEY_F5   = 7667712


# ── Detection worker ──────────────────────────────────────────────────────────

class _DetectionWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._lock      = threading.Lock()
        self._pending:  Optional[np.ndarray] = None
        self._preds:    list[dict] = []
        self._det_fps   = 0.0
        self._det_err   = ""
        self._new_frame = threading.Event()
        self._stop      = threading.Event()

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

class Eval2GUI:

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

        # Rollout state
        self._kb_home_set    = False
        self._rollout_idx    = 0
        self._score          = 0.0
        self._rollout_active = False
        self._all_done       = False
        self._rollout_start_t = 0.0

        # Per-rollout result log
        self._results: list[dict] = []

        self._status   = "Finding keyboard home …"
        self._show_pnp = True

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

    # ── KB_HOME ───────────────────────────────────────────────────────────────

    def _go_kb_home(self, then_start: bool = False) -> None:
        self._cancel()
        self._rollout_active = False
        with self._press_lock:
            self._cancel_evt.clear()
            t = threading.Thread(
                target=self._kb_home_worker, args=(then_start,), daemon=True
            )
            self._press_thread = t
        t.start()

    def _kb_home_worker(self, then_start: bool = False) -> None:
        self._status = "Finding keyboard home …"
        try:
            find_kb_home(self.robot, self.kin,
                         lambda: self._get_frame(fresh=True))
            self._kb_home_set = True
            self._status = "KB_HOME set — starting rollouts …"
        except Exception as exc:
            self._status = f"KB_HOME failed: {exc}"
            return
        if then_start:
            self._run_all_rollouts()

    # ── Reset home ────────────────────────────────────────────────────────────

    def _go_home(self) -> None:
        self._cancel()
        self._rollout_active = False
        with self._press_lock:
            self._cancel_evt.clear()
            t = threading.Thread(target=self._home_worker, daemon=True)
            self._press_thread = t
        t.start()

    def _home_worker(self) -> None:
        self._status = "Returning to reset home …"
        smooth_move(self.robot, self.kin, self._home)
        self._status = "At reset home — press [.] to re-find KB_HOME"

    # ── Cancel ────────────────────────────────────────────────────────────────

    def _cancel(self) -> None:
        self._cancel_evt.set()
        with self._press_lock:
            t = self._press_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._cancel_evt.clear()

    # ── Main rollout loop (runs inline in press_thread) ───────────────────────

    def _run_all_rollouts(self) -> None:
        for idx in range(self._rollout_idx, _ROLLOUT_COUNT):
            if self._cancel_evt.is_set():
                break

            key_label = TASK_SEQUENCE[idx]
            robot_label = _QWERTY_TO_QWERTZ.get(key_label, key_label)

            self._rollout_idx    = idx
            self._rollout_active = True
            self._rollout_start_t = time.monotonic()

            remaining_budget = _ROLLOUT_TIME_S
            self._status = (
                f"Rollout {idx+1}/{_ROLLOUT_COUNT}  key='{key_label}'"
                f"  score={self._score:.1f}  time={remaining_budget:.0f}s"
            )

            with self._press_lock:
                self._active_key = robot_label

            success = False
            try:
                press_key(
                    robot_label,
                    self.robot,
                    self.kin,
                    lambda: self._get_frame(fresh=True),
                    lift=False,
                    cancel_event=self._cancel_evt,
                )
                elapsed = time.monotonic() - self._rollout_start_t
                if not self._cancel_evt.is_set() and elapsed <= _ROLLOUT_TIME_S:
                    success = True
            except Exception as exc:
                self._status = f"Press error: {exc}"

            with self._press_lock:
                self._active_key = None

            elapsed = time.monotonic() - self._rollout_start_t
            if success:
                self._score += _POINTS_PER_ROLLOUT

            self._results.append({
                "rollout": idx + 1,
                "key": key_label,
                "success": success,
                "elapsed_s": round(elapsed, 2),
                "score_so_far": self._score,
            })

            status_icon = "✓" if success else "✗"
            self._status = (
                f"Rollout {idx+1} {status_icon}  key='{key_label}'"
                f"  elapsed={elapsed:.1f}s  score={self._score:.1f}"
            )
            print(
                f"[Eval2] rollout {idx+1:02d}/{_ROLLOUT_COUNT}  "
                f"key='{key_label}'  {'OK' if success else 'FAIL'}  "
                f"{elapsed:.1f}s  score={self._score:.1f}"
            )

            if self._cancel_evt.is_set():
                break

            # Return to KB_HOME between rollouts (not after the last one)
            if idx < _ROLLOUT_COUNT - 1:
                return_to_kb_home(self.robot)

        self._rollout_active = False
        self._all_done = True
        self._rollout_idx = _ROLLOUT_COUNT
        self._print_summary()
        self._status = (
            f"All done!  Final score: {self._score:.1f} / "
            f"{_ROLLOUT_COUNT * _POINTS_PER_ROLLOUT:.1f}  — press Esc to quit"
        )

    def _print_summary(self) -> None:
        total = _ROLLOUT_COUNT * _POINTS_PER_ROLLOUT
        n_ok  = sum(1 for r in self._results if r["success"])
        print(f"\n{'─'*50}")
        print(f"  Eval 2 Summary")
        print(f"  Rollouts : {len(self._results)}")
        print(f"  Correct  : {n_ok}/{len(self._results)}")
        print(f"  Score    : {self._score:.2f} / {total:.2f}")
        print(f"{'─'*50}")
        for r in self._results:
            icon = "✓" if r["success"] else "✗"
            print(f"  {r['rollout']:2d}. {icon}  '{r['key']}'  {r['elapsed_s']:.1f}s")
        print(f"{'─'*50}\n")

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

        # ── Top panel ─────────────────────────────────────────────────────────
        # Current target key (large)
        if self._rollout_active and active_key is not None:
            elapsed   = time.monotonic() - self._rollout_start_t
            remaining = max(0.0, _ROLLOUT_TIME_S - elapsed)
            key_txt   = active_key.upper()
            cv2.putText(frame, f"PRESS: {key_txt}", (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2, cv2.LINE_AA)
            # Timer
            t_color = (0, 60, 255) if remaining < 3.0 else (200, 255, 200)
            (tw, _), _ = cv2.getTextSize(f"{remaining:.1f}s", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, f"{remaining:.1f}s", (w - tw - 10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, t_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "─", (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)

        # Score + rollout progress
        score_str = (
            f"Rollout {min(self._rollout_idx+1, _ROLLOUT_COUNT)}/{_ROLLOUT_COUNT}"
            f"   Score: {self._score:.1f} / {_ROLLOUT_COUNT*_POINTS_PER_ROLLOUT:.1f}"
        )
        cv2.putText(frame, score_str, (8, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 80), 1, cv2.LINE_AA)

        # Sequence preview (show next few keys)
        preview_keys = TASK_SEQUENCE[self._rollout_idx:self._rollout_idx + 8]
        preview_str  = "  ".join(
            f"[{k.upper()}]" if i == 0 and self._rollout_active else k.upper()
            for i, k in enumerate(preview_keys)
        )
        cv2.putText(frame, preview_str, (8, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 255), 1, cv2.LINE_AA)

        # Result dots for completed rollouts
        dot_x, dot_y = 8, 100
        for i, r in enumerate(self._results):
            color = (60, 220, 60) if r["success"] else (60, 60, 220)
            cv2.circle(frame, (dot_x + i * 18, dot_y), 6, color, -1, cv2.LINE_AA)

        # Det stats
        n_keys   = sum(1 for p in preds if p["class"].lower() != "keyboard")
        pnp_flag = "PnP ON" if self._show_pnp else "PnP off"
        if det_err:
            cv2.putText(frame, f"DET ERR: {det_err[:55]}", (8, 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 80, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(
                frame,
                f"det {det_fps:.1f} Hz  |  {n_keys} key(s)  |  {pnp_flag}",
                (8, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1, cv2.LINE_AA,
            )

        # ── Status bar ────────────────────────────────────────────────────────
        bar_h = 46
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, self._status, (8, h - bar_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f",=home  .=KB_home  `=overlay  Esc=quit",
            (8, h - bar_h + 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 220, 160), 1, cv2.LINE_AA,
        )

        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

        seq_display = " ".join(
            k if k == "space" else k for k in TASK_SEQUENCE
        )
        print(f"{'─'*55}")
        print("  Eval 2 — a-z key press  (16 rollouts × 3.125 pts)")
        print(f"  Sequence (seed={_SENTENCE_SEED}): {seq_display}")
        print(f"  Time per rollout: {_ROLLOUT_TIME_S}s")
        print(f"{'─'*55}\n")

        # Auto-start: find KB_HOME then run all rollouts
        self._go_kb_home(then_start=True)

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

            if ch == 27:                       # Esc
                break
            elif ch == ord(","):
                self._go_home()
            elif ch == ord("."):
                self._go_kb_home(then_start=False)
            elif key == _KEY_HOME:
                self._go_home()
            elif key == _KEY_F5:
                self._go_kb_home(then_start=False)
            elif ch == 96:                     # `
                self._show_pnp = not self._show_pnp
                self._status = f"PnP overlay {'ON' if self._show_pnp else 'OFF'}"

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

    app = Eval2GUI(robot, kin, cap, home_pos=home)
    try:
        app.run()
    finally:
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
