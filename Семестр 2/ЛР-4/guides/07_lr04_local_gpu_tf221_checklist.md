# ЛР04 (Autoregression) — Local GPU Checklist (TensorFlow 2.21)

Цель: быстро поднять локальный GPU-контур для `ЛР04` и подтвердить запуск как минимум одного показательного `solution`-ноутбука на актуальном TensorFlow.

## 0) Что считаем «актуальным стеком»
Для `RTX 50xx` в этом курсе рабочий recovery-путь:
- `tensorflow[and-cuda]==2.21.0`
- `nvidia-cuda-nvcc-cu12>=12.8`

> Базовые `requirements.txt` в теме оставлены более широкими/консервативными для совместимости, но для локального GPU кейса `ЛР04` используем стек выше.

## 1) Preflight железа
- [ ] Проверить, что драйвер и GPU видны:

```bash
nvidia-smi -L
nvidia-smi
```

Ожидается видимый `GPU 0`.

## 2) Подготовить `.venv` проекта
Из корня репозитория:

- [ ] Создать окружение (если ещё нет):
```bash
python3 -m venv .venv
```

- [ ] Обновить pip и поставить базовые зависимости темы:
```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r themes/04-Autoregression/lab/requirements.txt
```

## 3) Переключить TensorFlow на локальный GPU-стек
- [ ] Установить TF2.21 + CUDA toolchain-пакет:

```bash
.venv/bin/python -m pip install --upgrade 'tensorflow[and-cuda]==2.21.0' 'nvidia-cuda-nvcc-cu12>=12.8'
```

- [ ] Проверить TensorFlow:

```bash
.venv/bin/python - <<'PY'
import tensorflow as tf
print('tf_version =', tf.__version__)
print('built_with_cuda =', tf.test.is_built_with_cuda())
PY
```

Ожидается `tf_version = 2.21.0`, `built_with_cuda = True`.

## 4) Зарегистрировать kernel для Jupyter
- [ ]
```bash
.venv/bin/python -m ipykernel install --user --name students-ai-lr04-gpu-tf221 --display-name 'Python (.venv) LR04 GPU TF2.21'
```

## 5) Запустить показательный `solution`-ноутбук из ЛР04
- [ ] Нейнтерактивный прогон GPU `solution` через скрипт:

```bash
GPU_TRAINING_BUDGET_MINUTES=5 \
  themes/04-Autoregression/lab/scripts/execute_gpu_solution.sh \
  themes/04-Autoregression/lab/solutions/runs_tf221_local_gpu_demo
```

## 6) Критерий PASS
Проверяем файл:

`themes/04-Autoregression/lab/solutions/runs_tf221_local_gpu_demo/02_decoder_only_tiny_shakespeare_gpu_solution.summary.json`

Минимальные условия:
- [ ] `overall_pass = true`
- [ ] `generation_pass = true`
- [ ] `baseline_pass = true`
- [ ] `success_count >= 19`

## 7) Если снова `INVALID_PTX` / `INVALID_HANDLE`
- [ ] Не переписывать модель и не менять математику ЛР.
- [ ] Перезапустить kernel (`Restart & Run All`) и повторить preflight.
- [ ] Проверить, что запускается именно GPU-ноутбук `02_decoder_only_tiny_shakespeare_gpu_solution.ipynb` (там есть `gpu_preflight()` и предзагрузка CUDA-библиотек из `.venv`).
- [ ] При нестабильности использовать fallback: `local-cpu`, `colab-gpu` или `kaggle-gpu`.

См. также:
- `05_tiny_shakespeare_gpu_walkthrough.md`
- `06_tiny_shakespeare_gpu_debugging_playbook.md`
- `../../../00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md`
