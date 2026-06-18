#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
JUPYTER_BIN="${REPO_ROOT}/.venv/bin/jupyter"
NOTEBOOK_PATH="${REPO_ROOT}/themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare_gpu.ipynb"

if [[ ! -x "${JUPYTER_BIN}" ]]; then
  echo "[ERROR] Не найден ${JUPYTER_BIN}." >&2
  echo "Сначала создайте окружение .venv и установите зависимости курса." >&2
  exit 1
fi

if [[ ! -f "${NOTEBOOK_PATH}" ]]; then
  echo "[ERROR] Не найдена тетрадь ${NOTEBOOK_PATH}." >&2
  exit 1
fi

export COURSE_RUNTIME_MODE=local-gpu

echo "[INFO] COURSE_RUNTIME_MODE=${COURSE_RUNTIME_MODE}"
echo "[INFO] Запуск GPU starter: ${NOTEBOOK_PATH}"

exec "${JUPYTER_BIN}" notebook "${NOTEBOOK_PATH}" "$@"
