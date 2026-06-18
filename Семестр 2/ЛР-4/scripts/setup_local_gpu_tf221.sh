#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Не найден ${PYTHON_BIN}." >&2
  echo "[HINT] Сначала создайте окружение: python3 -m venv .venv" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "[INFO] GPU setup for TensorFlow 2.21 (local RTX 50xx path)"
echo "[INFO] Repo root: ${REPO_ROOT}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] nvidia-smi -L"
  nvidia-smi -L || true
else
  echo "[WARN] nvidia-smi не найден в PATH."
fi

echo "[INFO] Обновляем pip"
"${PYTHON_BIN}" -m pip install --upgrade pip

echo "[INFO] Ставим базовые зависимости ЛР04 (без tensorflow-пина из requirements.txt)"
REQ_FILE="${REPO_ROOT}/themes/04-Autoregression/lab/requirements.txt"
TMP_REQ="$(mktemp)"
trap 'rm -f "${TMP_REQ}"' EXIT

grep -Ev '^[[:space:]]*tensorflow([[:space:]]*[<>=!~].*)?$' "${REQ_FILE}" > "${TMP_REQ}"
"${PYTHON_BIN}" -m pip install -r "${TMP_REQ}"

echo "[INFO] Применяем расширенный recovery для RTX 50xx"
"${PYTHON_BIN}" -m pip install --upgrade 'tensorflow[and-cuda]==2.21.0' 'nvidia-cuda-nvcc-cu12>=12.8'

echo "[INFO] Регистрируем Jupyter kernel"
"${PYTHON_BIN}" -m ipykernel install --user --name students-ai-gpu-tf221 --display-name 'Python (.venv) GPU TF2.21'

echo "[INFO] Проверяем версию TensorFlow"
"${PYTHON_BIN}" - <<'PY'
import tensorflow as tf
print('tf_version =', tf.__version__)
print('built_with_cuda =', tf.test.is_built_with_cuda())
PY

echo "[OK] Готово. Для показательного запуска используйте:"
echo "  GPU_TRAINING_BUDGET_MINUTES=5 themes/04-Autoregression/lab/scripts/execute_gpu_solution.sh themes/04-Autoregression/lab/solutions/runs_g635_demo"
