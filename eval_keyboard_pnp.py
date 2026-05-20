#!/usr/bin/env python3
"""
eval_keyboard_pnp.py — Policy evaluation for SO-101 keyboard pressing.

Drop-in evaluation counterpart to keyboard_press_gui_pnp.py.
Instead of waiting for mouse / keyboard input, the policy decides which key
to press next based on the live camera frame and the current task goal.

Controls (same as keyboard_press_gui_pnp.py)
--------------------------------------------
    Right-click / Esc   abort current episode
    HOME key            return to reset home
    F5                  re-find keyboard home (PnP)
    ` (backtick)        toggle PnP overlay
    p                   pause / resume policy
    Esc                 quit eval

Usage
-----
    python eval_keyboard_pnp.py \\
        --checkpoint outputs/checkpoints/last \\
        --target-text "hello" \\
        --episodes 10 \\
        --output-dir outputs/eval_runs
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime
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

# Physical QWERTY → QWERTZ (German layout target keyboard)
_QWERTY_TO_QWERTZ: dict[str, str] = {"y": "z", "z": "y"}

_WIN      = "SO-101  Keyboard Eval  [PnP]"
_KEY_HOME = 2359296   # VK_HOME
_KEY_F5   = 7667712   # VK_F5


# ══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER — Policy loader
# ══════════════════════════════════════════════════════════════════════════════

class PolicyPlaceholder:
    """
    Replace this class with your real policy (e.g. SmolVLA, Pi0, ACT …).

    The eval loop calls:
        key_label = policy.predict(frame, task_goal)

    where:
        frame      — (H, W, 3) uint8 BGR image from the wrist camera
        task_goal  — str, e.g. "type hello"

    Returns:
        key_label  — str matching a Roboflow class name (e.g. "a", "enter"),
                     or None to wait / do nothing this step.
    """

    def __init__(self, checkpoint_path: Optional[str], device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        # TODO: load weights, build model, move to device
        print(f"[Policy] Placeholder — checkpoint={checkpoint_path}  device={device}")
        print("[Policy] Replace PolicyPlaceholder with your real policy class.")

    def predict(self, frame: np.ndarray, task_goal: str) -> Optional[str]:
        # TODO: preprocess frame → tensor, run forward pass, decode action → key label
        return None   # placeholder: does nothing


# ══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER — Episode / task definition
# ══════════════════════════════════════════════════════════════════════════════

class EpisodePlaceholder:
    """
    Replace with real episode logic (goal generation, success detection, …).

    The eval loop calls:
        episode.reset()               → task_goal: str
        episode.step(key_pressed)     → (done: bool, success: bool, info: dict)
    """

    def __init__(self, target_text: str) -> None:
        self.target_text = target_text
        self._typed: list[str] = []

    def reset(self) -> str:
        self._typed = []
        return f"type: {self.target_text}"

    def step(self, key_pressed: Optional[str]) -> tuple[bool, bool, dict]:
        if key_pressed is None:
            return False, False, {}
        self._typed.append(key_pressed)
        typed_str = "".join(self._typed)
        success = typed_str.lower() == self.target_text.lower()
        # TODO: replace with real success / failure detection
        done = success or len(self._typed) >= len(self.target_text) + 3
        return done, success, {"typed": typed_str, "target": self.target_text}


# ══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER — Metrics / result logger
# ══════════════════════════════════════════════════════════════════════════════

class EvalLogger:
    """Saves per-episode results to a JSON file."""

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = output_dir / f"eval_{ts}.json"
        self._results: list[dict] = []

    def log_episode(self, ep_idx: int, success: bool, info: dict, duration_s: float) -> None:
        record = {"episode": ep_idx, "success": success, "duration_s": round(duration_s, 2), **info}
        self._results.append(record)
        self.path.write_text(json.dumps(self._results, indent=2))
        status = "✓ SUCCESS" if success else "✗ FAIL"
        print(f"[Eval] ep {ep_idx:03d}  {status}  {duration_s:.1f}s  {info}")

    def summary(self) -> None:
        n = len(self._results)
        if n == 0:
            print("[Eval] No episodes completed.")
            return
        successes = sum(1 for r in self._results if r["success"])
        print(f"\n[Eval] ── Summary ──────────────────")
        print(f"[Eval]   Episodes : {n}")
        print(f"[Eval]   Success  : {successes}/{n}  ({100*successes/n:.1f}%)")
        print(f"[Eval]   Results  : {self.path}")


# ══════════════════════════════════════════════════════════════════════════════
# Detection worker (same as keyboard_press_gui_pnp.py)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Eval GUI
# ══════════════════════════════════════════════════════════════════════════════

class KeyboardEvalGUI:

    def __init__(
        self,
        robot: SO101Follower,
        kin: SO101Kinematics,
        cap: cv2.VideoCapture,
        home_pos: np.ndarray,
        policy: PolicyPlaceholder,
        episode: EpisodePlaceholder,
        logger: EvalLogger,
        n_episodes: int,
        policy_hz: float = 2.0,
    ) -> None:
        self.robot      = robot
        self.kin        = kin
        self.cap        = cap
        self._home      = home_pos.copy()
        self.policy     = policy
        self.episode    = episode
        self.logger     = logger
        self.n_episodes = n_episodes
        self._policy_dt = 1.0 / max(policy_hz, 0.1)

        self._frame_lock  = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ts    = 0.0

        self._press_lock   = threading.Lock()
        self._press_thread: Optional[threading.Thread] = None
        self._cancel_evt   = threading.Event()
        self._active_key:  Optional[str] = None

        self._status       = "Ready  [PnP eval mode]"
        self._show_pnp     = True
        self._paused       = False

        self._ep_idx       = 0
        self._task_goal    = ""
        self._ep_start_t   = 0.0
        self._last_policy_t = 0.0

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
        if event == cv2.EVENT_RBUTTONDOWN:
            self._cancel()
            self._status = "Move cancelled (right-click)"

    # ── Press / cancel / home ─────────────────────────────────────────────────

    def _start_press(self, key_label: str) -> None:
        with self._press_lock:
            if self._press_thread is not None and self._press_thread.is_alive():
                return
        self._cancel()
        with self._press_lock:
            self._active_key = key_label
            self._cancel_evt.clear()
            t = threading.Thread(target=self._press_worker, args=(key_label,), daemon=True)
            self._press_thread = t
        t.start()

    def _press_worker(self, key_label: str) -> None:
        self._status = f"[PnP] Pressing '{key_label}' …"
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
            t = threading.Thread(target=lambda: smooth_move(self.robot, self.kin, self._home), daemon=True)
            self._press_thread = t
        t.start()
        self._status = "Returning to reset home …"

    def _go_kb_home(self) -> None:
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
            self._status = (f"At keyboard home  "
                            f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m")
        except Exception as exc:
            self._status = f"KB_HOME failed: {exc}"

    # ── Policy step ───────────────────────────────────────────────────────────

    def _is_pressing(self) -> bool:
        with self._press_lock:
            return self._press_thread is not None and self._press_thread.is_alive()

    def _policy_step(self) -> None:
        """Called from the main loop at policy_hz when not paused."""
        if self._is_pressing():
            return   # wait for the current press to finish

        frame = self._get_frame()
        if frame is None:
            return

        key_label = self.policy.predict(frame, self._task_goal)

        if key_label is not None:
            # QWERTY → QWERTZ mapping (same as keyboard_press_gui_pnp.py)
            robot_label = _QWERTY_TO_QWERTZ.get(key_label, key_label)
            self._start_press(robot_label)

            done, success, info = self.episode.step(key_label)
            if done:
                duration = time.monotonic() - self._ep_start_t
                self.logger.log_episode(self._ep_idx, success, info, duration)
                self._ep_idx += 1
                if self._ep_idx >= self.n_episodes:
                    self._status = "All episodes done — press Esc to quit"
                    return
                self._status = f"Episode {self._ep_idx}/{self.n_episodes} — go home and press F5 to continue"
                self._paused = True   # pause between episodes
        else:
            self._status = (f"[ep {self._ep_idx+1}/{self.n_episodes}]  "
                            f"goal: {self._task_goal!r}  |  policy waiting …")

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

        # ── Top-left stats ────────────────────────────────────────────────────
        n_keys   = sum(1 for p in preds if p["class"].lower() != "keyboard")
        pnp_flag = "PnP ON" if self._show_pnp else "PnP off"
        pause_flag = "PAUSED" if self._paused else f"ep {self._ep_idx+1}/{self.n_episodes}"
        if det_err:
            cv2.putText(frame, f"DET ERR: {det_err[:55]}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 80, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(
                frame,
                f"det {det_fps:.1f} Hz  |  {n_keys} key(s)  |  {pnp_flag}  |  {pause_flag}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 200), 1, cv2.LINE_AA,
            )

        # ── Status bar ────────────────────────────────────────────────────────
        bar_h = 50
        frame[h - bar_h:] = (frame[h - bar_h:] * 0.35).astype(np.uint8)
        cv2.putText(frame, self._status, (8, h - bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"p=pause  right-click=cancel  ,=home  .=KB_home  `=overlay  Esc=quit  "
            f"observe={INTERMEDIATE_OFFSET_M*100:.0f}cm → hover={HOVER_OFFSET_M*100:.0f}cm",
            (8, h - bar_h + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 220, 160), 1, cv2.LINE_AA,
        )

        # ── Task goal overlay (top-right) ─────────────────────────────────────
        goal_txt = f"goal: {self._task_goal!r}"
        (gw, gh), _ = cv2.getTextSize(goal_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, goal_txt, (w - gw - 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 80), 1, cv2.LINE_AA)

        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WIN, self._on_mouse)

        print(f"{'─' * 60}")
        print("  SO-101 Keyboard Eval GUI  [PnP mode]")
        print(f"  Camera {CAMERA_INDEX}  {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"  Episodes: {self.n_episodes}")
        print(f"  [F5]   Go to keyboard home ({KB_HOME_HEIGHT_M*100:.0f} cm above keyboard centre)")
        print( "  [HOME] Return to reset home")
        print( "  [p]    Pause / resume policy")
        print( "  [` ]   Toggle PnP overlay  |  [Esc] Quit")
        print(f"{'─' * 60}\n")

        # Start first episode
        self._task_goal  = self.episode.reset()
        self._ep_start_t = time.monotonic()
        self._status     = f"Episode 1/{self.n_episodes}  goal: {self._task_goal!r}  — F5 to find KB home first"

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

            # ── Policy tick ───────────────────────────────────────────────────
            if (not self._paused
                    and self._ep_idx < self.n_episodes
                    and ts - self._last_policy_t >= self._policy_dt):
                self._last_policy_t = ts
                self._policy_step()

            cv2.imshow(_WIN, self._draw(frame.copy()))

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            ch = key & 0xFF

            if ch == 27:                          # Esc = quit
                break
            elif ch == ord(","):                  # , = reset home
                self._go_home()
            elif ch == ord("."):                  # . = keyboard home
                self._go_kb_home()
            elif key == _KEY_HOME:
                self._go_home()
            elif key == _KEY_F5:
                self._go_kb_home()
            elif ch == 96:                        # ` = PnP overlay
                self._show_pnp = not self._show_pnp
                self._status = f"PnP overlay {'ON' if self._show_pnp else 'OFF'}"
            elif ch == ord("p"):                  # p = pause/resume
                self._paused = not self._paused
                if not self._paused and self._ep_idx < self.n_episodes:
                    self._ep_start_t = time.monotonic()
                    self._task_goal  = self.episode.reset()
                    self._status = f"Resumed — ep {self._ep_idx+1}/{self.n_episodes}  goal: {self._task_goal!r}"
                else:
                    self._status = "Paused — press [p] to resume next episode"

        self._cancel()
        self._det.stop()
        self.logger.summary()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-101 keyboard press eval (PnP mode)")
    p.add_argument("--checkpoint",   default=None,           help="Policy checkpoint path")
    p.add_argument("--device",       default="cpu",          help="Torch device (cpu / cuda)")
    p.add_argument("--target-text",  default="hello",        help="Text the robot should type per episode")
    p.add_argument("--episodes",     type=int, default=5,    help="Number of eval episodes")
    p.add_argument("--policy-hz",    type=float, default=2., help="Policy inference rate (Hz)")
    p.add_argument("--output-dir",   default="outputs/eval", help="Directory for result JSON files")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

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

    # ── Instantiate placeholders — swap in your real implementations ──────────
    policy  = PolicyPlaceholder(checkpoint_path=args.checkpoint, device=args.device)
    episode = EpisodePlaceholder(target_text=args.target_text)
    logger  = EvalLogger(output_dir=Path(args.output_dir))

    app = KeyboardEvalGUI(
        robot=robot,
        kin=kin,
        cap=cap,
        home_pos=home,
        policy=policy,
        episode=episode,
        logger=logger,
        n_episodes=args.episodes,
        policy_hz=args.policy_hz,
    )
    try:
        app.run()
    finally:
        cap.release()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
