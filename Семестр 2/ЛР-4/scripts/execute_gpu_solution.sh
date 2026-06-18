#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
JUPYTER_BIN="${REPO_ROOT}/.venv/bin/jupyter"
NOTEBOOK_PATH="${REPO_ROOT}/themes/04-Autoregression/lab/solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb"

DEFAULT_OUTPUT_DIR="${TMPDIR:-/tmp}/edu-ai-math-essentials-notebook-runs"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
OUTPUT_NAME="02_decoder_only_tiny_shakespeare_gpu_solution.executed.ipynb"
SUMMARY_NAME="02_decoder_only_tiny_shakespeare_gpu_solution.summary.json"
ATTEMPT_PROFILES=("gpu_60m" "gpu_60m_boost")

print_usage() {
  cat <<'USAGE'
Использование:
  execute_gpu_solution.sh [OUTPUT_DIR]

Описание:
  Выполняет GPU solution-тетрадь через jupyter nbconvert в режиме local-gpu
  и сохраняет выполненную копию в OUTPUT_DIR.

Аргументы:
  OUTPUT_DIR   Каталог для результата (по умолчанию:
               ${TMPDIR:-/tmp}/edu-ai-math-essentials-notebook-runs)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  echo "[ERROR] Ожидается не более одного аргумента: OUTPUT_DIR." >&2
  print_usage >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  OUTPUT_DIR="$1"
fi

if [[ ! -x "${JUPYTER_BIN}" ]]; then
  echo "[ERROR] Не найден ${JUPYTER_BIN}." >&2
  echo "Сначала создайте окружение .venv и установите зависимости курса." >&2
  exit 1
fi

if [[ ! -f "${NOTEBOOK_PATH}" ]]; then
  echo "[ERROR] Не найдена тетрадь ${NOTEBOOK_PATH}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export COURSE_RUNTIME_MODE=local-gpu
export GPU_TRAINING_BUDGET_MINUTES="${GPU_TRAINING_BUDGET_MINUTES:-60}"
export GPU_WARMUP_STEPS="${GPU_WARMUP_STEPS:-2}"
export GPU_WARMUP_PROBE_STEPS="${GPU_WARMUP_PROBE_STEPS:-2}"

echo "[INFO] COURSE_RUNTIME_MODE=${COURSE_RUNTIME_MODE}"
echo "[INFO] Сценарий: warm-up + timed-run"
echo "[INFO] Измеряемый бюджет после warm-up: ${GPU_TRAINING_BUDGET_MINUTES} минут"
echo "[INFO] Выполнение GPU solution через nbconvert"
echo "[INFO] Входная тетрадь: ${NOTEBOOK_PATH}"
echo "[INFO] Целевой выход: ${OUTPUT_DIR}/${OUTPUT_NAME}"

attempt_no=0
overall_pass=0
last_output_path=""

for profile in "${ATTEMPT_PROFILES[@]}"; do
  attempt_no=$((attempt_no + 1))
  export GPU_PROFILE_NAME="${profile}"
  attempt_output="02_decoder_only_tiny_shakespeare_gpu_solution.attempt${attempt_no}.${profile}.executed.ipynb"
  attempt_summary="02_decoder_only_tiny_shakespeare_gpu_solution.attempt${attempt_no}.${profile}.summary.json"
  attempt_output_path="${OUTPUT_DIR}/${attempt_output}"
  attempt_summary_path="${OUTPUT_DIR}/${attempt_summary}"
  last_output_path="${attempt_output_path}"

  echo "[INFO] ===== Attempt ${attempt_no}/${#ATTEMPT_PROFILES[@]} | profile=${GPU_PROFILE_NAME} ====="

  set +e
  "${JUPYTER_BIN}" nbconvert \
    --to notebook \
    --execute \
    --allow-errors \
    --ExecutePreprocessor.timeout=-1 \
    "${NOTEBOOK_PATH}" \
    --output "${attempt_output}" \
    --output-dir "${OUTPUT_DIR}"
  nbconvert_exit=$?
  set -e

  if [[ ${nbconvert_exit} -ne 0 ]]; then
    echo "[WARN] nbconvert завершился с кодом ${nbconvert_exit}. Пытаемся разобрать сохранённый результат."
  fi

  set +e
  python3 - "${attempt_output_path}" "${attempt_summary_path}" <<'PY'
import json
import sys
from pathlib import Path


def extract_text_chunks(output_obj):
    chunks = []
    if "text" in output_obj:
        text_value = output_obj["text"]
        if isinstance(text_value, list):
            chunks.extend(text_value)
        else:
            chunks.append(str(text_value))
    data = output_obj.get("data", {})
    if "text/plain" in data:
        plain_value = data["text/plain"]
        if isinstance(plain_value, list):
            chunks.extend(plain_value)
        else:
            chunks.append(str(plain_value))
    return chunks


def main(nb_path: Path, summary_path: Path) -> int:
    if not nb_path.is_file():
        print(f"[ERROR] Не найден выполненный ноутбук: {nb_path}")
        return 2

    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    summary_payload = None
    first_error = None

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error" and first_error is None:
                first_error = {
                    "ename": output.get("ename", ""),
                    "evalue": output.get("evalue", ""),
                }
            for chunk in extract_text_chunks(output):
                for line in chunk.splitlines():
                    if line.startswith("GPU_RUN_SUMMARY_JSON="):
                        summary_payload = line.split("=", 1)[1].strip()

    if summary_payload is None:
        print("[ERROR] В выполненном ноутбуке не найден GPU_RUN_SUMMARY_JSON.")
        if first_error is not None:
            print(
                f"[ERROR] Первая ошибка в ноутбуке: "
                f"{first_error['ename']}: {first_error['evalue']}"
            )
        return 3

    try:
        summary = json.loads(summary_payload)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Не удалось распарсить GPU_RUN_SUMMARY_JSON: {exc}")
        return 4

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[INFO] Сводка запуска:")
    print(
        f"  profile={summary.get('profile_name')} "
        f"stop_reason={summary.get('stop_reason')} "
        f"success={summary.get('success_count')}/{summary.get('prompt_count')} "
        f"mean={summary.get('mean_match_ratio'):.4f} "
        f"test_ppl={summary.get('test_perplexity'):.4f} "
        f"baseline_ppl={summary.get('baseline_perplexity'):.4f} "
        f"cpu_ref_pass={summary.get('cpu_reference_pass')}"
    )

    if bool(summary.get("overall_pass")):
        print("[INFO] PASS: жёсткий критерий 19/20 и базовый критерий перплексии достигнуты.")
        return 0

    print("[WARN] FAIL: целевые критерии не достигнуты на текущем профиле.")
    return 10


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("[ERROR] Internal parser usage: parser.py <notebook> <summary>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
PY
  parser_exit=$?
  set -e

  if [[ ${parser_exit} -eq 0 ]]; then
    cp "${attempt_output_path}" "${OUTPUT_DIR}/${OUTPUT_NAME}"
    cp "${attempt_summary_path}" "${OUTPUT_DIR}/${SUMMARY_NAME}"
    echo "[INFO] Финальная тетрадь: ${OUTPUT_DIR}/${OUTPUT_NAME}"
    echo "[INFO] Финальная сводка: ${OUTPUT_DIR}/${SUMMARY_NAME}"
    overall_pass=1
    break
  fi

  echo "[WARN] Профиль ${GPU_PROFILE_NAME} не прошёл критерии (код ${parser_exit})."
done

if [[ ${overall_pass} -ne 1 ]]; then
  echo "[ERROR] Не удалось достичь PASS после всех профилей." >&2
  echo "[ERROR] Последний выполненный ноутбук: ${last_output_path}" >&2
  exit 1
fi
