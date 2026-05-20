#!/usr/bin/env python3
"""
eval_keyboard_eval3.py — Eval 3 + Bonus: type an arbitrary sentence (a-z, space).

Scoring (per rollout):
    max(0, 5 - d)  points  after  10 * len(s) seconds
    where d = Levenshtein distance of typed sequence to input sentence
    and   len(s) = length of sentence including spaces (max 15).

Bonus: total wall-clock time across all eval-3 rollouts (lower = better).

Usage
-----
    # Headless (default):
    python eval_keyboard_eval3.py --sentence "hello world"

    # GUI overlay:
    python eval_keyboard_eval3.py --sentence "hello world" --gui

    # Interactive: script asks for sentence on stdin each time
    python eval_keyboard_eval3.py --interactive

The script types as fast as possible.  Return-to-KB_HOME happens between every
character so the camera can re-detect from the top-down view for the next press.
"""

import argparse
import os
import threading
import time
import warnings
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

_QWERTY_TO_QWERTZ: dict[str, str] = {"y": "z", "z": "y"}
_WIN = "SO-101  Eval 3  [sentence]"
_KEY_HOME = 2359296
_KEY_F5   = 7667712


# ── Levenshtein distance ──────────────────────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


# ── Sentence normalisation ────────────────────────────────────────────────────

def normalise_sentence(s: str) -> str:
    """Keep only a-z and spaces; collapse multiple spaces; strip."""
    cleaned = "".join(c if c.isalpha() or c == " " else "" for c in s.lower())
    return " ".join(cleaned.split())


def sentence_to_keys(sentence: str) -> list[str]:
    """Convert normalised sentence to list of key labels for press_key()."""
    keys = []
    for ch in sentence:
        if ch == " ":
            keys.append("space")
        else:
            keys.append(ch)
    return keys


# ── Frame store ───────────────────────────────────────────────────────────────

class _FrameStore:
    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts    = 0.0
        self._stop  = threading.Event()

    def start(self, cap: cv2.VideoCapture) -> None:
        def _reader() -> None:
            while not self._stop.is_set():
                ret, frame = cap.read()
                if ret:
                    ts = time.monotonic()
                    with self._lock:
                        self._frame = frame.copy()
                        self._ts    = ts
        threading.Thread(target=_reader, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def get(self, fresh: bool = False) -> Optional[np.ndarray]:
        if not fresh:
            with self._lock:
                return self._frame.copy() if self._frame is not None else None
        deadline = time.monotonic()
        while True:
            with self._lock:
                if self._ts > deadline and self._frame is not None:
                    return self._frame.copy()
            time.sleep(0.005)


# ── Core: type a sentence ─────────────────────────────────────────────────────

def type_sentence(
    sentence: str,
    robot: SO101Follower,
    kin: SO101Kinematics,
    get_frame,
    time_limit: float,
) -> tuple[str, float]:
    """
    Type sentence as fast as possible within time_limit seconds.
    Returns (typed_string, elapsed_seconds).
    """
    keys     = sentence_to_keys(sentence)
    typed    = []
    cancel   = threading.Event()
    deadline = time.monotonic() + time_limit
    t_start  = time.monotonic()

    for idx, key in enumerate(keys):
        now = time.monotonic()
        if now >= deadline:
            print(f"  [timeout] stopping at char {idx+1}/{len(keys)}", flush=True)
            break

        robot_label = _QWERTY_TO_QWERTZ.get(key, key)
        remaining   = deadline - now
        print(f"  [{idx+1:2d}/{len(keys)}] '{key}'  {remaining:.1f}s left …", flush=True)

        cancel.clear()
        success = False
        _t_key = time.monotonic()
        try:
            press_key(robot_label, robot, kin,
                      lambda: get_frame(fresh=True),
                      lift=False, cancel_event=cancel)
            if not cancel.is_set():
                success = True
        except Exception as exc:
            print(f"    [error] {exc}", flush=True)
        _t_after_press = time.monotonic()
        print(f"  [TIMING] char {idx+1} '{key}' press_key wall: {_t_after_press-_t_key:.2f}s", flush=True)

        if success:
            typed.append(sentence[idx] if key != "space" else " ")
        else:
            typed.append("?")   # mark as failed but keep position

        if cancel.is_set():
            break

        # Return to KB_HOME between chars (not after the last one)
        if idx < len(keys) - 1:
            return_to_kb_home(robot)
        print(f"  [TIMING] char {idx+1} '{key}' TOTAL (press+return): {time.monotonic()-_t_key:.2f}s", flush=True)

    elapsed = time.monotonic() - t_start
    # Keep '?' for failed presses — counted as a substitution in Levenshtein
    # so each failure costs 1 edit distance (not silently removed).
    typed_str = "".join(typed)
    return typed_str, elapsed


# ── Headless runner ───────────────────────────────────────────────────────────

def run_headless(
    robot: SO101Follower,
    kin: SO101Kinematics,
    cap: cv2.VideoCapture,
    sentences: list[str],
) -> None:
    store = _FrameStore()
    store.start(cap)
    try:
        print("Finding keyboard home …", flush=True)
        _t_home = time.monotonic()
        find_kb_home(robot, kin, lambda: store.get(fresh=True))
        print(f"KB_HOME set.  [TIMING] find_kb_home: {time.monotonic()-_t_home:.2f}s\n", flush=True)

        total_time = 0.0
        results: list[dict] = []

        for run_idx, sentence in enumerate(sentences):
            sentence    = normalise_sentence(sentence)
            time_limit  = 10.0 * len(sentence)
            max_pts     = 5

            print(f"{'─'*50}")
            print(f"Rollout {run_idx+1}/{len(sentences)}")
            print(f"  Sentence : '{sentence}'")
            print(f"  Length   : {len(sentence)}  Time limit: {time_limit:.0f}s")
            print(f"{'─'*50}", flush=True)

            typed, elapsed = type_sentence(
                sentence, robot, kin, store.get, time_limit
            )

            d        = levenshtein(sentence, typed)
            points   = max(0, max_pts - d)
            n_failed = typed.count("?")
            # Bonus timing: if errors, penalise with full time_limit
            bonus_t = elapsed if d == 0 else time_limit
            total_time += bonus_t

            print(f"\n  Typed    : '{typed}'")
            print(f"  Target   : '{sentence}'")
            print(f"  Failed   : {n_failed} press(es) returned '?'")
            print(f"  Distance : {d}  →  {points}/{max_pts} pts")
            print(f"  Time     : {elapsed:.1f}s  (bonus time: {bonus_t:.1f}s)\n")

            results.append({
                "rollout":  run_idx + 1,
                "sentence": sentence,
                "typed":    typed,
                "distance": d,
                "points":   points,
                "elapsed_s": round(elapsed, 2),
                "bonus_t":   round(bonus_t, 2),
            })

            if run_idx < len(sentences) - 1:
                return_to_kb_home(robot)

        # Summary
        total_pts = sum(r["points"] for r in results)
        max_total = len(results) * 5
        print(f"{'═'*50}")
        print(f"  FINAL SCORE : {total_pts} / {max_total} pts")
        print(f"  BONUS TIME  : {total_time:.1f}s  (lower = better)")
        print(f"{'═'*50}")
        for r in results:
            d_icon = "✓" if r["distance"] == 0 else f"d={r['distance']}"
            print(f"  {r['rollout']:2d}. {d_icon:6s}  "
                  f"'{r['sentence']}'  →  '{r['typed']}'  "
                  f"{r['elapsed_s']:.1f}s  {r['points']}pts")
        print(f"{'═'*50}\n")
    finally:
        store.stop()


# ── GUI runner ────────────────────────────────────────────────────────────────

def run_gui(
    robot: SO101Follower,
    kin: SO101Kinematics,
    cap: cv2.VideoCapture,
    sentences: list[str],
) -> None:
    frame_lock    = threading.Lock()
    latest: list[Optional[np.ndarray]] = [None]
    frame_ts: list[float] = [0.0]
    cancel        = threading.Event()

    def _get_frame(fresh: bool = False) -> Optional[np.ndarray]:
        if not fresh:
            with frame_lock:
                return latest[0].copy() if latest[0] is not None else None
        dl = time.monotonic()
        while True:
            with frame_lock:
                if frame_ts[0] > dl and latest[0] is not None:
                    return latest[0].copy()
            time.sleep(0.005)

    state = {
        "status": "Finding keyboard home …",
        "active_key": None,
        "sentence": "",
        "typed": "",
        "char_idx": 0,
        "n_chars": 0,
        "deadline": 0.0,
        "show_pnp": True,
        "done": False,
    }

    from keyboard_pnp import detections_from_roboflow, draw_pnp_overlay

    det_worker = None
    try:
        from eval_keyboard_eval2 import _DetectionWorker
        det_worker = _DetectionWorker()
        det_worker.start()
    except Exception:
        pass

    def _worker() -> None:
        state["status"] = "Finding keyboard home …"
        find_kb_home(robot, kin, lambda: _get_frame(fresh=True))
        state["status"] = "KB_HOME set"

        for run_idx, sentence in enumerate(sentences):
            sentence   = normalise_sentence(sentence)
            time_limit = 10.0 * len(sentence)
            keys       = sentence_to_keys(sentence)
            state["sentence"] = sentence
            state["n_chars"]  = len(keys)
            state["typed"]    = ""
            state["deadline"] = time.monotonic() + time_limit
            typed = []

            for idx, key in enumerate(keys):
                if time.monotonic() >= state["deadline"] or cancel.is_set():
                    break
                state["char_idx"]  = idx
                state["active_key"] = key
                state["status"] = (
                    f"[{run_idx+1}/{len(sentences)}] char {idx+1}/{len(keys)}: '{key}'"
                )
                robot_label = _QWERTY_TO_QWERTZ.get(key, key)
                cancel.clear()
                try:
                    press_key(robot_label, robot, kin,
                              lambda: _get_frame(fresh=True),
                              lift=False, cancel_event=cancel)
                    typed.append(sentence[idx] if key != "space" else " ")
                except Exception as exc:
                    state["status"] = f"Error: {exc}"
                state["active_key"] = None
                state["typed"] = "".join(c for c in typed if c != "?")
                if not cancel.is_set() and idx < len(keys) - 1:
                    return_to_kb_home(robot)

            typed_str = "".join(c for c in typed if c != "?")
            d = levenshtein(sentence, typed_str)
            state["status"] = (
                f"Done: '{typed_str}'  d={d}  pts={max(0,5-d)}"
            )
            if run_idx < len(sentences) - 1:
                return_to_kb_home(robot)

        state["done"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        ts = time.monotonic()
        with frame_lock:
            latest[0]  = frame.copy()
            frame_ts[0] = ts

        if det_worker is not None:
            det_worker.post_frame(frame)
            preds, det_fps, det_err = det_worker.get_state()
            if state["show_pnp"] and preds:
                dets = detections_from_roboflow(preds)
                if dets:
                    draw_pnp_overlay(frame, dets, CAMERA_K, DIST_COEFFS)

        h, w = frame.shape[:2]

        # Sentence progress
        s   = state["sentence"]
        idx = state["char_idx"]
        if s:
            done_part = s[:idx]
            cur_char  = s[idx] if idx < len(s) else ""
            rest      = s[idx+1:] if idx + 1 < len(s) else ""
            prog      = f"{done_part}[{cur_char}]{rest}"
            cv2.putText(frame, prog, (8, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)

        # Timer
        remaining = max(0.0, state["deadline"] - time.monotonic())
        tc = (0, 60, 255) if remaining < 5 else (200, 255, 200)
        (tw, _), _ = cv2.getTextSize(f"{remaining:.1f}s",
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, f"{remaining:.1f}s", (w - tw - 10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tc, 2, cv2.LINE_AA)

        # Typed so far
        cv2.putText(frame, f"typed: '{state['typed']}'", (8, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 255, 180), 1, cv2.LINE_AA)

        bar_h = 40
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, state["status"], (8, h - bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "`=overlay  Esc=quit",
                    (8, h - bar_h + 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, (160, 220, 160), 1, cv2.LINE_AA)

        cv2.imshow(_WIN, frame)
        k = cv2.waitKeyEx(1)
        if k != -1:
            ch = k & 0xFF
            if ch == 27:
                cancel.set()
                break
            elif ch == 96:
                state["show_pnp"] = not state["show_pnp"]

        if state["done"]:
            time.sleep(3)
            break

    cancel.set()
    if det_worker:
        det_worker.stop()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval 3 — type a sentence (a-z, space) on the keyboard"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--sentence", metavar="S",
                       help="Sentence to type, e.g. 'hello world'")
    group.add_argument("--interactive", action="store_true",
                       help="Ask for sentence on stdin each rollout")
    p.add_argument("--rollouts", type=int, default=1,
                   help="Number of rollouts in interactive mode (default 1)")
    p.add_argument("--gui", action="store_true", default=False,
                   help="Show OpenCV window")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not ROBOFLOW_API_KEY:
        raise RuntimeError(f"ROBOFLOW_API_KEY not set — add it to {_ENV_FILE}")

    # Collect sentences
    sentences: list[str] = []
    if args.sentence:
        sentences = [args.sentence]
    elif args.interactive:
        print(f"Enter {args.rollouts} sentence(s), one per line (a-z + space):")
        for i in range(args.rollouts):
            try:
                s = input(f"  Sentence {i+1}: ")
            except EOFError:
                break
            sentences.append(s)
    else:
        print("No sentence provided. Use --sentence 'hello world' or --interactive.")
        return

    sentences = [normalise_sentence(s) for s in sentences if s.strip()]
    if not sentences:
        print("No valid sentences after normalisation.")
        return

    print(f"{'─'*55}")
    print(f"  Eval 3  {'(GUI)' if args.gui else '(headless)'}")
    for i, s in enumerate(sentences):
        tl = 10.0 * len(s)
        print(f"  [{i+1}] '{s}'  len={len(s)}  time_limit={tl:.0f}s")
    print(f"{'─'*55}\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kin = SO101Kinematics(URDF_PATH)

    robot = SO101Follower(SO101FollowerConfig(
        port=PORT, id=ROBOT_ID,
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

    try:
        if args.gui:
            run_gui(robot, kin, cap, sentences)
        else:
            run_headless(robot, kin, cap, sentences)
    finally:
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
