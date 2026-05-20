#!/usr/bin/env python3
"""
eval_keyboard_eval2.py — Eval 2: press a given a-z / space key within 10 s.
16 rollouts × 3.125 pts = 50 pts max.

The evaluator provides the character(s) — we do NOT use a fixed sequence.

Usage
-----
    # Interactive (default): evaluator types one key per rollout via stdin
    python eval_keyboard_eval2.py

    # Single key (evaluator calls once per rollout):
    python eval_keyboard_eval2.py --key a

    # Batch sequence (evaluator provides all keys upfront):
    python eval_keyboard_eval2.py --sequence "axihexdsnbcgh"

    # GUI overlay:
    python eval_keyboard_eval2.py --gui
    python eval_keyboard_eval2.py --sequence "abc" --gui

Timing
------
    Interactive: 10 s window starts when the key is received from stdin.
    Batch/single: 10 s window starts when arm receives the character input;
                  return-to-KB_HOME counts against the NEXT window (same as
                  the physical evaluation cadence).
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

# ── Constants ─────────────────────────────────────────────────────────────────
_ROLLOUT_TIME_S     = 10.0
_POINTS_PER_ROLLOUT = 3.125
_QWERTY_TO_QWERTZ: dict[str, str] = {"y": "z", "z": "y"}
_WIN      = "SO-101  Eval 2  [a-z]"
_KEY_HOME = 2359296
_KEY_F5   = 7667712


def _parse_key_input(raw: str) -> Optional[str]:
    """Normalise evaluator input → key label for press_key(), or None."""
    s = raw.strip().lower()
    if s in ("space", " ", "spc"):
        return "space"
    if s in ("enter", "return"):
        return "enter"
    if len(s) == 1 and s.isalpha():
        return s
    return None


# ── Frame / camera helpers ────────────────────────────────────────────────────

class _FrameStore:
    """Thread-safe latest-frame store, filled by a background reader."""

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


# ── Core press logic ──────────────────────────────────────────────────────────

def do_press(
    key_label: str,
    robot: SO101Follower,
    kin: SO101Kinematics,
    store: _FrameStore,
    deadline: float,
    cancel: threading.Event,
) -> tuple[bool, float]:
    """Press key_label; return (success, elapsed_s from window open)."""
    window_open = deadline - _ROLLOUT_TIME_S
    robot_label = _QWERTY_TO_QWERTZ.get(key_label, key_label)
    try:
        press_key(robot_label, robot, kin,
                  lambda: store.get(fresh=True),
                  lift=False, cancel_event=cancel)
        t = time.monotonic()
        success = not cancel.is_set() and t <= deadline
    except Exception as exc:
        t = time.monotonic()
        success = False
        print(f"  [error] {exc}", flush=True)
    elapsed = t - window_open
    return success, elapsed


# ── Headless runners ──────────────────────────────────────────────────────────

def _setup_kb_home(robot, kin, store):
    print("Finding keyboard home …", flush=True)
    find_kb_home(robot, kin, lambda: store.get(fresh=True))
    print("KB_HOME set.\n", flush=True)


def run_interactive(
    robot: SO101Follower,
    kin: SO101Kinematics,
    store: _FrameStore,
    n_rollouts: int,
) -> None:
    _setup_kb_home(robot, kin, store)

    results: list[dict] = []
    score = 0.0
    cancel = threading.Event()

    for idx in range(n_rollouts):
        # Arm is at KB_HOME — signal ready and wait for evaluator input
        print(f"[{idx+1:2d}/{n_rollouts}] READY — enter key: ", end="", flush=True)
        try:
            raw = input()
        except EOFError:
            break

        key = _parse_key_input(raw)
        if key is None:
            print(f"  [skip] unrecognised input '{raw}'", flush=True)
            continue

        # 10 s window starts NOW (when key is received)
        deadline = time.monotonic() + _ROLLOUT_TIME_S
        print(f"  pressing '{key}' …", flush=True)
        cancel.clear()

        success, elapsed = do_press(key, robot, kin, store, deadline, cancel)

        if success:
            score += _POINTS_PER_ROLLOUT
        icon = "✓" if success else "✗"
        print(f"  {icon} {elapsed:.2f}s | score={score:.1f}", flush=True)
        results.append({"rollout": idx + 1, "key": key,
                        "success": success, "elapsed_s": round(elapsed, 2)})

        if idx < n_rollouts - 1:
            return_to_kb_home(robot)

    _print_summary(results, score)


def run_single_key(
    robot: SO101Follower,
    kin: SO101Kinematics,
    store: _FrameStore,
    key: str,
) -> None:
    _setup_kb_home(robot, kin, store)
    cancel   = threading.Event()
    deadline = time.monotonic() + _ROLLOUT_TIME_S
    print(f"Pressing '{key}' …", flush=True)
    success, elapsed = do_press(key, robot, kin, store, deadline, cancel)
    icon = "✓" if success else "✗"
    print(f"{icon} {elapsed:.2f}s", flush=True)


def run_sequence(
    robot: SO101Follower,
    kin: SO101Kinematics,
    store: _FrameStore,
    sequence: list[str],
) -> None:
    _setup_kb_home(robot, kin, store)

    results: list[dict] = []
    score    = 0.0
    cancel   = threading.Event()
    # First window opens now
    deadline = time.monotonic() + _ROLLOUT_TIME_S

    for idx, key in enumerate(sequence):
        window_open = deadline - _ROLLOUT_TIME_S
        time_left   = max(0.0, deadline - time.monotonic())
        print(f"[{idx+1:2d}/{len(sequence)}] '{key}'  budget={time_left:.1f}s  pressing …",
              flush=True)
        cancel.clear()

        success, elapsed = do_press(key, robot, kin, store, deadline, cancel)

        press_time = time.monotonic()   # deadline for NEXT key starts here
        if success:
            score += _POINTS_PER_ROLLOUT
        icon = "✓" if success else "✗"
        print(f"  {icon} {elapsed:.2f}s | score={score:.1f}", flush=True)
        results.append({"rollout": idx + 1, "key": key,
                        "success": success, "elapsed_s": round(elapsed, 2)})

        if cancel.is_set():
            break

        next_deadline = press_time + _ROLLOUT_TIME_S
        if idx < len(sequence) - 1:
            return_to_kb_home(robot)
        deadline = next_deadline

    _print_summary(results, score)


def _print_summary(results: list[dict], score: float) -> None:
    total = len(results) * _POINTS_PER_ROLLOUT
    n_ok  = sum(1 for r in results if r["success"])
    print(f"\n{'─'*45}")
    print(f"  Eval 2 — score: {score:.2f} / {total:.2f}  ({n_ok}/{len(results)} correct)")
    print(f"{'─'*45}")
    for r in results:
        icon = "✓" if r["success"] else "✗"
        print(f"  {r['rollout']:2d}. {icon}  '{r['key']}'  {r['elapsed_s']:.2f}s")
    print(f"{'─'*45}\n")


# ── GUI mode (optional) ───────────────────────────────────────────────────────

class _DetectionWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._lock      = threading.Lock()
        self._pending:  Optional[np.ndarray] = None
        self._preds:    list[dict] = []
        self._det_fps   = 0.0
        self._det_err   = ""
        self._new_frame = threading.Event()
        self._stop_evt  = threading.Event()

    def post_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame
        self._new_frame.set()

    def get_state(self) -> tuple[list[dict], float, str]:
        with self._lock:
            return list(self._preds), self._det_fps, self._det_err

    def stop(self) -> None:
        self._stop_evt.set()
        self._new_frame.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
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
                    self._preds, self._det_fps, self._det_err = preds, fps, ""
            except Exception as exc:
                with self._lock:
                    self._det_err = str(exc)[:80]


def run_gui(
    robot: SO101Follower,
    kin: SO101Kinematics,
    cap: cv2.VideoCapture,
    home_pos: np.ndarray,
    sequence: Optional[list[str]],
    n_rollouts: int,
) -> None:
    """GUI wrapper: shows camera + detection overlay while running the sequence."""
    frame_lock   = threading.Lock()
    latest_frame: list[Optional[np.ndarray]] = [None]
    frame_ts:     list[float] = [0.0]

    def _get_frame(fresh: bool = False) -> Optional[np.ndarray]:
        if not fresh:
            with frame_lock:
                return latest_frame[0].copy() if latest_frame[0] is not None else None
        deadline = time.monotonic()
        while True:
            with frame_lock:
                if frame_ts[0] > deadline and latest_frame[0] is not None:
                    return latest_frame[0].copy()
            time.sleep(0.005)

    # Shared GUI state
    state = {
        "status": "Finding keyboard home …",
        "active_key": None,
        "rollout_idx": 0,
        "score": 0.0,
        "deadline": 0.0,
        "window_open": 0.0,
        "results": [],
        "show_pnp": True,
        "done": False,
    }

    det = _DetectionWorker()
    det.start()

    cancel  = threading.Event()
    press_thread: list[Optional[threading.Thread]] = [None]

    def _worker():
        try:
            state["status"] = "Finding keyboard home …"
            find_kb_home(robot, kin, lambda: _get_frame(fresh=True))
            state["status"] = "KB_HOME set"

            keys = sequence if sequence else []
            score = 0.0

            if sequence:
                deadline = time.monotonic() + _ROLLOUT_TIME_S
                for idx, key in enumerate(keys):
                    state["rollout_idx"] = idx
                    state["deadline"]    = deadline
                    state["window_open"] = deadline - _ROLLOUT_TIME_S
                    state["active_key"]  = key
                    state["status"] = f"[{idx+1}/{len(keys)}] pressing '{key}' …"
                    cancel.clear()

                    success, elapsed = do_press(
                        key, robot, kin,
                        type("S", (), {"get": staticmethod(_get_frame)})(),
                        deadline, cancel,
                    )
                    press_time = time.monotonic()
                    if success:
                        score += _POINTS_PER_ROLLOUT
                    state["score"] = score
                    state["active_key"] = None
                    icon = "✓" if success else "✗"
                    state["status"] = f"{icon} '{key}' {elapsed:.1f}s | score={score:.1f}"
                    state["results"].append({"key": key, "success": success})
                    if cancel.is_set():
                        break
                    next_deadline = press_time + _ROLLOUT_TIME_S
                    if idx < len(keys) - 1:
                        return_to_kb_home(robot)
                    deadline = next_deadline
            else:
                # Interactive: can't do GUI + blocking input simultaneously
                # Fall back to sequence display hint
                state["status"] = "Run without --gui for interactive mode. Use --sequence."

            state["done"] = True
            state["status"] = f"Done! Score={score:.1f}/{len(keys)*_POINTS_PER_ROLLOUT:.1f}"
            _print_summary(state["results_full"] if "results_full" in state else [], score)
        except Exception as exc:
            state["status"] = f"Error: {exc}"
            state["done"] = True

    t = threading.Thread(target=_worker, daemon=True)
    press_thread[0] = t
    t.start()

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        ts = time.monotonic()
        with frame_lock:
            latest_frame[0] = frame.copy()
            frame_ts[0]     = ts

        det.post_frame(frame)

        # Draw
        preds, det_fps, det_err = det.get_state()
        h, w = frame.shape[:2]
        if state["show_pnp"] and preds:
            dets = detections_from_roboflow(preds)
            if dets:
                draw_pnp_overlay(frame, dets, CAMERA_K, DIST_COEFFS)
        for p in preds:
            lbl = p["class"]
            px, py = float(p["x"]), float(p["y"])
            bw, bh = float(p["width"]), float(p["height"])
            x1, x2 = int(px - bw/2), int(px + bw/2)
            y1, y2 = int(py - bh/2), int(py + bh/2)
            color = (255, 80, 0) if lbl == state["active_key"] else (50, 220, 50)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Active key + timer
        ak = state["active_key"]
        if ak:
            remaining = max(0.0, state["deadline"] - time.monotonic())
            cv2.putText(frame, f"PRESS: {ak.upper()}", (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2, cv2.LINE_AA)
            tc = (0, 60, 255) if remaining < 3 else (200, 255, 200)
            (tw, _), _ = cv2.getTextSize(f"{remaining:.1f}s",
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, f"{remaining:.1f}s", (w - tw - 10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, tc, 2, cv2.LINE_AA)

        cv2.putText(frame,
                    f"Score: {state['score']:.1f}  Rollout: {state['rollout_idx']+1}",
                    (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 80), 1, cv2.LINE_AA)

        bar_h = 40
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, state["status"], (8, h - bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(_WIN, frame)
        key = cv2.waitKeyEx(1)
        if key != -1:
            ch = key & 0xFF
            if ch == 27:
                cancel.set()
                break
            elif ch == 96:
                state["show_pnp"] = not state["show_pnp"]

    cancel.set()
    det.stop()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval 2 — press a-z keys on demand (16 rollouts × 3.125 pts)"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--key", metavar="K",
                       help="Single key to press (a-z or 'space')")
    group.add_argument("--sequence", metavar="CHARS",
                       help="All keys as a string, e.g. 'axihexd' or 'a x i h e x d'")
    p.add_argument("--rollouts", type=int, default=16,
                   help="Number of rollouts in interactive mode (default 16)")
    p.add_argument("--gui", action="store_true", default=False,
                   help="Show OpenCV window")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not ROBOFLOW_API_KEY:
        raise RuntimeError(f"ROBOFLOW_API_KEY not set — add it to {_ENV_FILE}")

    print(f"{'─'*55}")
    print(f"  Eval 2  {'(GUI)' if args.gui else '(headless)'}  {_ROLLOUT_TIME_S}s per key")
    if args.key:
        print(f"  Mode: single key  →  '{args.key}'")
    elif args.sequence:
        seq_list = [_parse_key_input(c) or c
                    for c in (args.sequence.split() if " " in args.sequence
                              else list(args.sequence))]
        print(f"  Mode: sequence  →  {seq_list}")
    else:
        print(f"  Mode: interactive  ({args.rollouts} rollouts via stdin)")
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

    home = kin.forward_kinematics(HOME_DEG)[:3, 3].copy()

    try:
        if args.gui:
            seq: Optional[list[str]] = None
            if args.key:
                seq = [_parse_key_input(args.key) or args.key]
            elif args.sequence:
                raw_seq = (args.sequence.split() if " " in args.sequence
                           else list(args.sequence))
                seq = [_parse_key_input(c) or c for c in raw_seq]
            run_gui(robot, kin, cap, home, seq, args.rollouts)
        else:
            store = _FrameStore()
            store.start(cap)
            try:
                if args.key:
                    key = _parse_key_input(args.key)
                    if key is None:
                        raise ValueError(f"Invalid key: '{args.key}'")
                    run_single_key(robot, kin, store, key)
                elif args.sequence:
                    raw_seq = (args.sequence.split() if " " in args.sequence
                               else list(args.sequence))
                    seq_list = [_parse_key_input(c) or c for c in raw_seq]
                    run_sequence(robot, kin, store, seq_list)
                else:
                    run_interactive(robot, kin, store, args.rollouts)
            finally:
                store.stop()
    finally:
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
