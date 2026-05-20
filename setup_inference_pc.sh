#!/usr/bin/env bash
# setup_inference_pc.sh — One-shot environment setup for the inference PC.
#
# Run once on the remote Ubuntu machine:
#   bash setup_inference_pc.sh
#
# What it does:
#   1. Checks for Python 3.12 and uv
#   2. Creates a lerobot venv and installs the project (hardware + feetech + eval extras)
#   3. Sets up the Roboflow inference server venv (keyboard detection)
#   4. Creates .env.inference from the example if it doesn't exist yet
#   5. Verifies the camera and serial port are accessible
#
# After running this script, start the full eval with:
#   bash run_eval.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
KBD_DIR="${REPO_DIR}/keyboard_detection"
KBD_VENV="${KBD_DIR}/.venv312"
ENV_FILE="${KBD_DIR}/.env.inference"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
# ── Config — override via env vars before running ──────────────────────────────
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyUSB0}"   # serial port the SO-101 is on
CAMERA_INDEX="${CAMERA_INDEX:-1}"           # OpenCV camera index

# ── Colour helpers ─────────────────────────────────────────────────────────────
_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
_bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

_bold "─────────────────────────────────────────────────────"
_bold "  SO-101 Keyboard-Eval  —  Inference-PC Setup"
_bold "─────────────────────────────────────────────────────"
echo  "  Repo  : ${REPO_DIR}"
echo  "  Robot : ${ROBOT_PORT}"
echo  "  Camera: ${CAMERA_INDEX}"
echo

# ── Step 1: Python 3.12 ────────────────────────────────────────────────────────
_bold "[1/5] Checking Python 3.12 ..."
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    _red "  ${PYTHON_BIN} not found."
    echo  "  Install it first:"
    echo  "    sudo add-apt-repository ppa:deadsnakes/ppa"
    echo  "    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev"
    exit 1
fi
PY_VER=$("${PYTHON_BIN}" --version)
_green "  Found ${PY_VER}"

# ── Step 2: uv ────────────────────────────────────────────────────────────────
_bold "[2/5] Checking uv ..."
if ! command -v uv >/dev/null 2>&1; then
    _yellow "  uv not found — installing ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Reload PATH so uv is available in this session
    export PATH="${HOME}/.local/bin:${PATH}"
fi
_green "  uv $(uv --version)"

# ── Step 3: lerobot venv ───────────────────────────────────────────────────────
_bold "[3/5] Setting up lerobot venv (${VENV_DIR}) ..."
cd "${REPO_DIR}"
if [[ ! -d "${VENV_DIR}" ]]; then
    uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
fi

# Install lerobot with the extras needed for hardware + eval
# Adjust extras here if your policy requires additional packages (e.g. [pi], [smolvla])
uv pip install --python "${VENV_DIR}/bin/python" \
    -e ".[hardware,feetech,evaluation,dataset]" \
    opencv-python \
    ikpy \
    Pillow \
    requests

_green "  lerobot installed."

# ── Step 4: Roboflow inference server ─────────────────────────────────────────
_bold "[4/5] Setting up Roboflow inference server ..."
if [[ ! -d "${KBD_VENV}" ]]; then
    "${PYTHON_BIN}" -m venv "${KBD_VENV}"
fi
"${KBD_VENV}/bin/python" -m pip install --upgrade pip --quiet
"${KBD_VENV}/bin/pip" install inference-cli --quiet

mkdir -p \
    "${KBD_DIR}/.roboflow-cache" \
    "${KBD_DIR}/.huggingface-cache" \
    "${KBD_DIR}/.yolo-cache" \
    "${KBD_DIR}/.rf-home" \
    "${KBD_DIR}/.mplconfig" \
    "${KBD_DIR}/outputs"

if [[ ! -f "${ENV_FILE}" ]]; then
    if [[ -f "${KBD_DIR}/.env.inference.example" ]]; then
        cp "${KBD_DIR}/.env.inference.example" "${ENV_FILE}"
        _yellow "  Created ${ENV_FILE} from example."
        _yellow "  !! Open it and fill in ROBOFLOW_API_KEY before running eval !!"
    else
        cat > "${ENV_FILE}" <<'EOF'
ROBOFLOW_API_KEY=FILL_IN_YOUR_KEY_HERE
ROBOFLOW_MODEL_ID=keyboard-key-recognition-kw7nc/14
INFERENCE_PORT=9001
INFERENCE_HOST=http://localhost:9001
EOF
        _yellow "  Created ${ENV_FILE} with placeholder key — edit before running!"
    fi
else
    _green "  ${ENV_FILE} already exists."
fi
_green "  Roboflow inference server ready."

# ── Step 5: Hardware checks ────────────────────────────────────────────────────
_bold "[5/5] Hardware checks ..."

# Serial port
if [[ -e "${ROBOT_PORT}" ]]; then
    _green "  Serial port ${ROBOT_PORT} exists."
    if ! groups | grep -q dialout; then
        _yellow "  WARNING: your user is not in the 'dialout' group."
        _yellow "  Run:  sudo usermod -aG dialout \$USER  (then log out and back in)"
    fi
else
    _yellow "  Serial port ${ROBOT_PORT} NOT found — is the robot plugged in?"
fi

# Camera (quick check — actual open happens at runtime)
if ls /dev/video* >/dev/null 2>&1; then
    _green "  Camera devices: $(ls /dev/video* | tr '\n' ' ')"
else
    _yellow "  No /dev/video* devices found — is the camera connected?"
fi

echo
_bold "─────────────────────────────────────────────────────"
_green "  Setup complete."
echo  "  Next steps:"
echo  "    1. Edit ${ENV_FILE} and set ROBOFLOW_API_KEY"
echo  "    2. Edit eval_keyboard_pnp.py and set PORT / CAMERA_INDEX / checkpoint path"
echo  "    3. Run:  bash run_eval.sh"
_bold "─────────────────────────────────────────────────────"
