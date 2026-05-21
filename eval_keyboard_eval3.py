#!/usr/bin/env python3
"""
eval_keyboard_eval3.py — Eval 3: type arbitrary sentences (a-z + space).
10 rollouts × max(0, 5 − d) pts, where d = Levenshtein(typed, target).
Time per rollout: 10 × len(sentence) seconds  (len counts spaces, max 15).

Usage
-----
    python eval_keyboard_eval3.py
    python eval_keyboard_eval3.py --sentences sentences.txt

  sentences.txt  — plain text, one sentence per line (10 lines).
  If omitted, the placeholder sentences at the top of this file are used.

Flow (fully automatic)
----------------------
    1. Script launches → finds KB_HOME → starts rollout 1.
    2. Robot types each character of the sentence within the time budget.
    3. Returns to KB_HOME → next rollout.
    4. After 10 rollouts, prints final score.

Controls
--------
    Esc     Quit early
    ,       Return to reset home (cancels current rollout)
    .       Re-find keyboard home
    `       Toggle PnP overlay
"""

import argparse
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

_ROLLOUT_COUNT       = 10
_SECS_PER_CHAR       = 10.0   # budget = _SECS_PER_CHAR × len(sentence)
_MAX_PTS_PER_ROLLOUT = 5

# QWERTY → QWERTZ (German layout keyboard)
_QWERTY_TO_QWERTZ: dict[str, str] = {"y": "z", "z": "y"}

# Placeholder sentences — replace with actual evaluation sentences before running,
# or supply them via --sentences sentences.txt.
# Requirements: a-z + space only, len(sentence) <= 15.
_DEFAULT_SENTENCES: list[str] = [
    "hello world",
    "cat and dog",
    "fly over it",
    "open the box",
    "red blue sky",
    "move it now",
    "find the key",
    "big wave hit",
    "go left fast",
    "win the cup",
]

_WIN      = "SO-101  Eval 3  [sentence typing]"
_KEY_HOME = 2359296
_KEY_F5   = 7667712


# ── Helpers ───────────────────────────────────────────────────────────────────

def _levenshtein(s: str, t: str) -> int:
    """Standard DP Levenshtein distance."""
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s[i - 1] == t[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _sentence_to_keys(sentence: str) -> list[tuple[str, str]]:
    """Return list of (robot_key, display_char) pairs for the sentence."""
    result: list[tuple[str, str]] = []
    for ch in sentence:
        if ch == " ":
            result.append(("space", " "))
        elif "a" <= ch <= "z":
            robot_key = _QWERTY_TO_QWERTZ.get(ch, ch)
            result.append((robot_key, ch))
    return result


def _load_sentences(path: str) -> list[str]:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(lines) < _ROLLOUT_COUNT:
        raise ValueError(
            f"sentences file has {len(lines)} lines, need {_ROLLOUT_COUNT}"
        )
    return lines[:_ROLLOUT_COUNT]


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

class Eval3GUI:

    def __init__(
        self,
        robot: SO101Follower,
        kin: SO101Kinematics,
        cap: cv2.VideoCapture,
        home_pos: np.ndarray,
        sentences: list[str],
    ) -> None:
        self.robot      = robot
        self.kin        = kin
        self.cap        = cap
        self._home      = home_pos.copy()
        self._sentences = sentences

        self._frame_lock   = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ts     = 0.0

        self._press_lock   = threading.Lock()
        self._press_thread: Optional[threading.Thread] = None
        self._cancel_evt   = threading.Event()
        self._active_key:  Optional[str] = None

        # Rollout state
        self._kb_home_set     = False
        self._rollout_idx     = 0
        self._score           = 0.0
        self._rollout_active  = False
        self._all_done        = False
        self._rollout_start_t = 0.0
        self._rollout_time_s  = 0.0

        # Per-rollout typing state (read by _draw from main thread)
        self._typed_chars:      list[str] = []
        self._target_sentence:  str = ""
        self._char_idx:         int = 0

        self._n_rollouts   = len(sentences)
        self._eval_start_t = 0.0
        self._results: list[dict] = []
        self._status   = "Finding keyboard home …"
        self._show_pnp = True

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

    # ── Main rollout loop ─────────────────────────────────────────────────────

    def _run_all_rollouts(self) -> None:
        self._eval_start_t = time.monotonic()
        for idx in range(self._rollout_idx, self._n_rollouts):
            if self._cancel_evt.is_set():
                break

            sentence       = self._sentences[idx]
            key_seq        = _sentence_to_keys(sentence)
            rollout_time_s = _SECS_PER_CHAR * len(sentence)

            self._rollout_idx     = idx
            self._rollout_active  = True
            self._rollout_start_t = time.monotonic()
            self._rollout_time_s  = rollout_time_s
            self._typed_chars     = []
            self._target_sentence = sentence
            self._char_idx        = 0

            self._status = (
                f"Rollout {idx+1}/{self._n_rollouts}  "
                f"'{sentence}'  budget={rollout_time_s:.0f}s"
            )
            print(
                f"\n[Eval3] rollout {idx+1:02d}/{self._n_rollouts}  "
                f"sentence='{sentence}'  budget={rollout_time_s:.0f}s"
            )

            for ki, (robot_key, display_char) in enumerate(key_seq):
                if self._cancel_evt.is_set():
                    break
                elapsed = time.monotonic() - self._rollout_start_t
                if elapsed >= rollout_time_s:
                    break

                self._char_idx = ki
                _MAX_ATTEMPTS  = 3
                pressed        = False

                for attempt in range(_MAX_ATTEMPTS):
                    if self._cancel_evt.is_set():
                        break
                    if time.monotonic() - self._rollout_start_t >= rollout_time_s:
                        break

                    with self._press_lock:
                        self._active_key = robot_key

                    try:
                        press_key(
                            robot_key,
                            self.robot,
                            self.kin,
                            lambda: self._get_frame(fresh=True),
                            lift=False,
                            cancel_event=self._cancel_evt,
                        )
                        pressed = True
                    except Exception as exc:
                        self._status = (
                            f"Press error [{robot_key}] "
                            f"({attempt + 1}/{_MAX_ATTEMPTS}): {str(exc)[:50]}"
                        )
                        # Lift back to KB_HOME so the camera regains a clear
                        # top-down view and the serial link stays active.
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

                if pressed and not self._cancel_evt.is_set():
                    self._typed_chars.append(display_char)

            with self._press_lock:
                self._active_key = None

            typed_str = "".join(self._typed_chars)
            d         = _levenshtein(typed_str, sentence)
            pts       = max(0, _MAX_PTS_PER_ROLLOUT - d)
            self._score += pts
            elapsed = time.monotonic() - self._rollout_start_t

            self._results.append({
                "rollout":      idx + 1,
                "sentence":     sentence,
                "typed":        typed_str,
                "distance":     d,
                "points":       pts,
                "elapsed_s":    round(elapsed, 2),
                "score_so_far": self._score,
            })

            status_icon = "✓" if d == 0 else f"d={d}"
            self._status = (
                f"Rollout {idx+1} [{status_icon}]  "
                f"typed='{typed_str}'  pts={pts}  score={self._score:.1f}"
            )
            print(
                f"[Eval3] rollout {idx+1:02d}  target='{sentence}'  "
                f"typed='{typed_str}'  d={d}  pts={pts}  score={self._score:.1f}"
            )

            if self._cancel_evt.is_set():
                break

            if idx < self._n_rollouts - 1:
                return_to_kb_home(self.robot)

        self._rollout_active = False
        self._all_done       = True
        self._rollout_idx    = self._n_rollouts
        self._print_summary()
        self._status = (
            f"All done!  Final score: {self._score:.1f} / "
            f"{self._n_rollouts * _MAX_PTS_PER_ROLLOUT:.1f}  — press Esc to quit"
        )

    def _print_summary(self) -> None:
        total        = self._n_rollouts * _MAX_PTS_PER_ROLLOUT
        n_perfect    = sum(1 for r in self._results if r["distance"] == 0)
        total_elapsed = time.monotonic() - self._eval_start_t if self._eval_start_t else 0.0
        m, s = divmod(int(total_elapsed), 60)
        print(f"\n{'─'*65}")
        print(f"  Eval 3 Summary")
        print(f"  Rollouts   : {len(self._results)}")
        print(f"  Perfect    : {n_perfect}/{len(self._results)}")
        print(f"  Score      : {self._score:.2f} / {total:.2f}")
        print(f"  Total time : {m}m {s:02d}s  ({total_elapsed:.1f}s)")
        print(f"{'─'*65}")
        for r in self._results:
            print(
                f"  {r['rollout']:2d}. pts={r['points']}  d={r['distance']}  "
                f"'{r['sentence']}' → '{r['typed']}'  {r['elapsed_s']:.1f}s"
            )
        print(f"{'─'*65}\n")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        preds, det_fps, det_err = self._det.get_state()

        with self._press_lock:
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
        y_off = 8

        if self._rollout_active:
            elapsed   = time.monotonic() - self._rollout_start_t
            remaining = max(0.0, self._rollout_time_s - elapsed)

            # Target sentence with current char highlighted
            target = self._target_sentence
            typed  = "".join(self._typed_chars)
            ci     = self._char_idx

            tgt_display = ""
            for i, ch in enumerate(target):
                if i < len(typed):
                    tgt_display += ch.upper()          # already typed
                elif i == ci:
                    tgt_display += f"[{ch.upper()}]"   # current target
                else:
                    tgt_display += ch.upper()

            cv2.putText(frame, f"TYPE: {tgt_display}", (8, y_off + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2, cv2.LINE_AA)

            # Typed so far
            typed_display = f"    > {typed}" if typed else "    > (waiting)"
            cv2.putText(frame, typed_display, (8, y_off + 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)

            # Timer
            t_color = (0, 60, 255) if remaining < 15.0 else (200, 255, 200)
            (tw, _), _ = cv2.getTextSize(f"{remaining:.1f}s", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, f"{remaining:.1f}s", (w - tw - 10, y_off + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, t_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "─", (8, y_off + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)

        # Score + rollout progress
        score_str = (
            f"Rollout {min(self._rollout_idx+1, self._n_rollouts)}/{self._n_rollouts}"
            f"   Score: {self._score:.1f} / {self._n_rollouts * _MAX_PTS_PER_ROLLOUT:.1f}"
        )
        cv2.putText(frame, score_str, (8, y_off + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 80), 1, cv2.LINE_AA)

        # Result dots for completed rollouts
        dot_y = y_off + 90
        for i, r in enumerate(self._results):
            color = (60, 220, 60) if r["distance"] == 0 else (
                (60, 200, 200) if r["points"] > 0 else (60, 60, 220)
            )
            cv2.circle(frame, (8 + i * 20, dot_y), 7, color, -1, cv2.LINE_AA)
            cv2.putText(frame, str(r["points"]), (4 + i * 20, dot_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1, cv2.LINE_AA)

        # Det stats
        n_keys   = sum(1 for p in preds if p["class"].lower() != "keyboard")
        pnp_flag = "PnP ON" if self._show_pnp else "PnP off"
        if det_err:
            cv2.putText(frame, f"DET ERR: {det_err[:55]}", (8, dot_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 80, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(
                frame,
                f"det {det_fps:.1f} Hz  |  {n_keys} key(s)  |  {pnp_flag}",
                (8, dot_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1, cv2.LINE_AA,
            )

        # ── Status bar ────────────────────────────────────────────────────────
        bar_h = 46
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, self._status, (8, h - bar_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            ",=home  .=KB_home  `=overlay  Esc=quit",
            (8, h - bar_h + 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 220, 160), 1, cv2.LINE_AA,
        )

        return frame

    # ── Shared header ─────────────────────────────────────────────────────────

    def _print_header(self) -> None:
        n = self._n_rollouts
        print(f"{'─'*60}")
        print(f"  Eval 3 — sentence typing  ({n} rollout{'s' if n != 1 else ''} × max 5 pts)")
        print(f"  Scoring  : max(0, 5 - Levenshtein(typed, target))")
        print(f"  Budget   : {_SECS_PER_CHAR:.0f}s × len(sentence) per rollout")
        print(f"  Sentences:")
        for i, s in enumerate(self._sentences):
            print(f"    {i+1:2d}. '{s}'  (len={len(s)}, budget={_SECS_PER_CHAR*len(s):.0f}s)")
        print(f"{'─'*60}\n")

    # ── Headless run (no GUI window) ──────────────────────────────────────────

    def run_headless(self) -> None:
        self._print_header()
        print("[headless] Starting — no display window")
        self._go_kb_home(then_start=True)
        try:
            while not self._all_done:
                ret, frame = self.cap.read()
                if ret:
                    ts = time.monotonic()
                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                        self._frame_ts     = ts
                    self._det.post_frame(frame)
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n[headless] Interrupted by user")
            self._cancel()
        finally:
            self._det.stop()

    # ── GUI main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        self._print_header()
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

            if ch == 27:
                break
            elif ch == ord(","):
                self._go_home()
            elif ch == ord("."):
                self._go_kb_home(then_start=False)
            elif key == _KEY_HOME:
                self._go_home()
            elif key == _KEY_F5:
                self._go_kb_home(then_start=False)
            elif ch == 96:
                self._show_pnp = not self._show_pnp
                self._status = f"PnP overlay {'ON' if self._show_pnp else 'OFF'}"

        self._cancel()
        self._det.stop()
        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Eval 3: sentence typing")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sentence", metavar="SENTENCE",
        help="single sentence to type (1 rollout, max 5 pts)",
    )
    group.add_argument(
        "--sentences", metavar="FILE",
        help="text file with 10 sentences, one per line (10 rollouts, max 50 pts)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="run without an OpenCV display window",
    )
    args = parser.parse_args()

    if not ROBOFLOW_API_KEY:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not set.\n"
            f"Add it to  {_ENV_FILE}"
        )

    if args.sentence:
        sentences = [args.sentence.strip().lower()]
    elif args.sentences:
        sentences = _load_sentences(args.sentences)
    else:
        sentences = _DEFAULT_SENTENCES
        print("[warn] Using placeholder sentences — supply --sentences FILE for actual eval")

    # Validate
    for i, s in enumerate(sentences):
        if not s:
            raise ValueError(f"Sentence {i+1} is empty")
        bad = [c for c in s if c != " " and not ("a" <= c <= "z")]
        if bad:
            raise ValueError(f"Sentence {i+1} '{s}' contains invalid chars: {bad}")
        if len(s) > 15:
            raise ValueError(f"Sentence {i+1} '{s}' length {len(s)} > 15")

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

    app = Eval3GUI(robot, kin, cap, home_pos=home, sentences=sentences)
    try:
        if args.headless:
            app.run_headless()
        else:
            app.run()
    finally:
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
