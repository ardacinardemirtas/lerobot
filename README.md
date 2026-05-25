# SO-101 Keyboard-Pressing Eval

An extension of [LeRobot](https://github.com/huggingface/lerobot) that programs an SO-101 robotic arm to autonomously press keys on a physical keyboard using real-time computer vision.

## System overview

| Component | Details |
|-----------|---------|
| Robot | SO-101 6-DOF follower arm |
| Vision | USB camera + Roboflow inference server (keyboard key detection) |
| Localization | Iterative PnP (`cv2.solvePnP`) to map pixel detections to 3-D key positions |
| Motion | QP-based inverse kinematics (`move_to_position_qp.py`) |
| Base library | [LeRobot](https://github.com/huggingface/lerobot) (HuggingFace) |

The robot detects keys in the camera frame, solves for their 3-D position relative to the end-effector, then executes smooth joint-space trajectories to reach and press each key.

## Eval tasks

### Eval 1 — Sequential key press (`eval_keyboard_task1_seq.py`)
Press **Space → Enter → R → L** in order within **40 seconds**.  
Scoring: 12.5 pts per correctly pressed key — **50 pts max**.

### Eval 2 — Single key on demand (`eval_keyboard_eval2.py`)
Robot is given a random a–z character and must press it within **10 seconds**.  
16 rollouts × 3.125 pts — **50 pts max**.

### Eval 3 — Sentence typing (`eval_keyboard_eval3.py`)
Robot types full sentences (a–z + space) from a provided list.  
10 rollouts, scored by `max(0, 5 − Levenshtein(typed, target))` — **50 pts max**.

## Repository layout

```
├── eval_keyboard_task1_seq.py   # Eval 1 runner
├── eval_keyboard_eval2.py       # Eval 2 runner
├── eval_keyboard_eval3.py       # Eval 3 runner
├── run_eval_1.sh                # Install + run Eval 1
├── run_eval_2.sh                # Install + run Eval 2
├── run_eval_3.sh                # Install + run Eval 3
├── setup_inference_pc.sh        # One-time environment setup
├── keyboard_detection/          # Roboflow detection helpers + venv
├── click_to_move.py             # Camera & robot constants
├── move_to_position_qp.py       # QP inverse kinematics
├── press_key_pnp.py             # Key-press primitives (PnP + motion)
├── keyboard_pnp.py              # PnP overlay utilities
├── sentences.txt                # Sentences used by Eval 3
└── src/lerobot/                 # Upstream LeRobot library
```

## Setup

### Prerequisites

- Ubuntu 22.04 or 24.04 (tested); Python 3.12; `uv` (installed automatically if missing)
- SO-101 arm connected via USB serial
- USB camera connected (default index 1)
- [Roboflow](https://roboflow.com) account — you need a **ROBOFLOW_API_KEY**

### One-time environment install

```bash
# Clone the repo
git clone <this-repo-url>
cd lerobot

# Run the setup script (installs lerobot venv + Roboflow inference server)
bash setup_inference_pc.sh

# Fill in your API key
nano keyboard_detection/.env.inference   # set ROBOFLOW_API_KEY=...
```

The setup script will:
1. Create `.venv` with LeRobot + hardware extras
2. Create `keyboard_detection/.venv312` with the Roboflow inference CLI
3. Write a template `keyboard_detection/.env.inference` if one does not exist

### Hardware constants

Open `click_to_move.py` and verify these match your setup:

```python
PORT         = "/dev/ttyUSB0"   # SO-101 serial port
CAMERA_INDEX = 1                # OpenCV camera index
ROBOT_ID     = "my_so101"      # calibration directory name
```

### Calibration

If you have not calibrated the arm yet:

```bash
.venv/bin/lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=my_so101
```

## Running the evals

Each script installs the required environment (if not already done) then starts the Roboflow inference server and launches the eval GUI.

```bash
bash run_eval_1.sh   # Eval 1 — Space → Enter → R → L
bash run_eval_2.sh   # Eval 2 — single key on demand
bash run_eval_3.sh   # Eval 3 — sentence typing
```

Pass `--sentences <file>` to `run_eval_3.sh` to use a custom sentence list:

```bash
bash run_eval_3.sh --sentences sentences.txt
```

### In-GUI controls (all evals)

| Key | Action |
|-----|--------|
| `.` | Find keyboard home (PnP — do this first) |
| `s` / `Space` | Start the timed run |
| `,` or `Home` | Return to reset home |
| `` ` `` | Toggle PnP overlay |
| `Esc` | Quit |

## Citation

This project builds on LeRobot:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and others},
    title  = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    year   = {2024},
    url    = {https://github.com/huggingface/lerobot}
}
```
