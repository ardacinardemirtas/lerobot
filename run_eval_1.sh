#!/usr/bin/env bash
# run_eval_1.sh — Install environment (if needed) then run Eval 1.
#
# Eval 1: press Space → Enter → R → L in sequence within 40 seconds.
# Scoring: 12.5 pts per key, 50 pts max.
#
# Usage:
#   bash run_eval_1.sh [extra args forwarded to eval_keyboard_task1_seq.py]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KBD_DIR="${REPO_DIR}/keyboard_detection"
KBD_VENV="${KBD_DIR}/.venv312"
ENV_FILE="${KBD_DIR}/.env.inference"
VENV_DIR="${REPO_DIR}/.venv"

INFERENCE_PORT="${INFERENCE_PORT:-9001}"

# ── Auto-install if venvs are missing ─────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]] || [[ ! -d "${KBD_VENV}" ]]; then
    echo "[run_eval_1] Environments not found — running setup_inference_pc.sh ..."
    bash "${REPO_DIR}/setup_inference_pc.sh"
fi

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found. Run setup_inference_pc.sh first." >&2
    exit 1
fi

if [[ ! -x "${KBD_VENV}/bin/inference" ]]; then
    echo "ERROR: Roboflow inference CLI not found. Run setup_inference_pc.sh first." >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: lerobot venv not found at ${VENV_DIR}. Run setup_inference_pc.sh first." >&2
    exit 1
fi

# ── Load env ───────────────────────────────────────────────────────────────────
set -a
source "${ENV_FILE}"
set +a

: "${ROBOFLOW_API_KEY:?Set ROBOFLOW_API_KEY in ${ENV_FILE}}"

# ── Tmp dirs for the inference server ─────────────────────────────────────────
for d in /tmp/model-cache /tmp/huggingface /tmp/yolo /tmp/matplotlib /tmp/home; do
    if [[ -L "${d}" ]]; then rm -f "${d}"; fi
    mkdir -p "${d}"
done

# ── Start inference server in background ───────────────────────────────────────
echo "[run_eval_1] Starting Roboflow inference server on port ${INFERENCE_PORT} ..."
"${KBD_VENV}/bin/inference" server start \
    --port "${INFERENCE_PORT}" \
    --use-local-images &
INFERENCE_PID=$!

_cleanup() {
    echo
    echo "[run_eval_1] Stopping inference server (PID ${INFERENCE_PID}) ..."
    kill "${INFERENCE_PID}" 2>/dev/null || true
    wait "${INFERENCE_PID}" 2>/dev/null || true
    echo "[run_eval_1] Done."
}
trap _cleanup EXIT INT TERM

# Wait until the server is ready (up to 30 s)
echo "[run_eval_1] Waiting for inference server to be ready ..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${INFERENCE_PORT}/info" >/dev/null 2>&1; then
        echo "[run_eval_1] Inference server ready."
        break
    fi
    if (( i == 30 )); then
        echo "ERROR: Inference server did not start in 30 s." >&2
        exit 1
    fi
    sleep 1
done

# ── Run Eval 1 ─────────────────────────────────────────────────────────────────
echo "[run_eval_1] Starting Eval 1: Space → Enter → R → L ..."
"${VENV_DIR}/bin/python" "${REPO_DIR}/eval_keyboard_task1_seq.py" "$@"
