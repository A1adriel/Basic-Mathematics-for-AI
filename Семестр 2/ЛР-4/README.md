# Лабораторный практикум по авторегрессионному моделированию

## Назначение
Практикум включает обязательный трёхшаговый маршрут внутри темы `04`:
1. `ЛР01`: декодерный трансформер (decoder-only Transformer) на детерминированной синтетике.
2. `ЛР02 CPU`: перенос на реальный корпус `Tiny Shakespeare` в варианте для центрального процессора (CPU).
3. `ЛР02 GPU`: расширенный перенос на тот же корпус в отдельном `GPU`-ноутбуке.

Цель блока — сначала закрепить механику причинной маски (causal mask), затем подтвердить прикладную переносимость на реальных данных, и после этого масштабировать тот же подход в отдельном вычислительном контуре.

## Структура
- `01_decoder_only_causal_toy.ipynb` — стартовая тетрадь `ЛР01`.
- `02_decoder_only_tiny_shakespeare.ipynb` — стартовая тетрадь `ЛР02 CPU`.
- `02_decoder_only_tiny_shakespeare_gpu.ipynb` — стартовая тетрадь `ЛР02 GPU`.
- `solutions/01_decoder_only_causal_toy_solution.ipynb` — решение `ЛР01`.
- `solutions/02_decoder_only_tiny_shakespeare_solution.ipynb` — решение `ЛР02 CPU`.
- `solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb` — решение `ЛР02 GPU`.
- `guides/00_autoregression_prerequisites.md` — входной минимум.
- `guides/01_decoder_only_toy_walkthrough.md` — пошаговый разбор `ЛР01`.
- `guides/02_autoregression_debugging_playbook.md` — диагностика `ЛР01`.
- `guides/03_tiny_shakespeare_walkthrough.md` — пошаговый разбор `ЛР02 CPU`.
- `guides/04_tiny_shakespeare_debugging_playbook.md` — диагностика `ЛР02 CPU`.
- `guides/05_tiny_shakespeare_gpu_walkthrough.md` — пошаговый разбор `ЛР02 GPU`.
- `guides/06_tiny_shakespeare_gpu_debugging_playbook.md` — диагностика `ЛР02 GPU`.

## Карта курса
Общая линия курса:
1. Шаг 1 = `01-RNN / ЛР01`
2. Шаг 2 = `01-RNN / ЛР02`
3. Шаг 3 = `01-RNN / ЛР03`
4. Шаг 4 = `02-Attention / ЛР01`
5. Шаг 5 = `03-Transformer / ЛР01`
6. Шаг 6 = `03-Transformer / ЛР02`
7. Шаг 7 = `04-Autoregression / ЛР01`
8. Шаг 8 = `04-Autoregression / ЛР02 CPU`
9. Шаг 9 = `04-Autoregression / ЛР02 GPU`
10. Шаг 10 = `05-Full-Transformer / ЛР05`

## Канонический маршрут студента
1. Прочитать [guides/00_autoregression_prerequisites.md](./guides/00_autoregression_prerequisites.md).
2. Пройти [guides/01_decoder_only_toy_walkthrough.md](./guides/01_decoder_only_toy_walkthrough.md).
3. Выполнить [01_decoder_only_causal_toy.ipynb](./01_decoder_only_causal_toy.ipynb).
4. При затруднениях использовать [guides/02_autoregression_debugging_playbook.md](./guides/02_autoregression_debugging_playbook.md).
5. Свериться с [solutions/01_decoder_only_causal_toy_solution.ipynb](./solutions/01_decoder_only_causal_toy_solution.ipynb).
6. Пройти [guides/03_tiny_shakespeare_walkthrough.md](./guides/03_tiny_shakespeare_walkthrough.md).
7. Выполнить [02_decoder_only_tiny_shakespeare.ipynb](./02_decoder_only_tiny_shakespeare.ipynb).
8. При затруднениях использовать [guides/04_tiny_shakespeare_debugging_playbook.md](./guides/04_tiny_shakespeare_debugging_playbook.md).
9. Свериться с [solutions/02_decoder_only_tiny_shakespeare_solution.ipynb](./solutions/02_decoder_only_tiny_shakespeare_solution.ipynb).
10. Пройти [guides/05_tiny_shakespeare_gpu_walkthrough.md](./guides/05_tiny_shakespeare_gpu_walkthrough.md).
11. Выполнить [02_decoder_only_tiny_shakespeare_gpu.ipynb](./02_decoder_only_tiny_shakespeare_gpu.ipynb).
12. При затруднениях использовать [guides/06_tiny_shakespeare_gpu_debugging_playbook.md](./guides/06_tiny_shakespeare_gpu_debugging_playbook.md).
13. Свериться с [solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb](./solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb).

## Контракты выполнения `Tiny Shakespeare`
### `ЛР02 CPU` (обязательный)
- фиксированный `seed`;
- фиксированное индексное разбиение;
- детерминированная генерация через `argmax`;
- обязательное сравнение с частотным ориентиром.

### `ЛР02 GPU` (отдельный обязательный артефакт)
- отдельная стартовая тетрадь и отдельное решение;
- отдельные параметры данных и модели;
- обязательный отдельный этап `warm-up` для JIT/компиляции перед измеряемым обучением;
- запуск только в среде с доступным `GPU` и корректными CUDA-драйверами;
- измеряемый учебный бюджет обучения: `60` минут после warm-up;
- валидация каждые `cfg['eval_every_steps']` шагов;
- целевая генерация: не ниже `19/20` фиксированных подсказок;
- ранняя остановка (early stopping) по `val_loss` как вспомогательный контур.

### CPU/GPU + leakage contract (единый быстрый срез)
| Контур | Что обязательно | Что запрещено |
|---|---|---|
| `CPU` | детерминированный pipeline, baseline/perplexity/generation gates | скрытое изменение постановки |
| `GPU` | `gpu_preflight()` до обучения, отдельный warm-up, измеряемый бюджет | hidden CPU fallback при выбранном GPU-профиле |
| `Leakage` | causal mask и проверка будущей массы внимания | доступ decoder к будущим токенам |

`leakage` в этом блоке означает любую утечку информации о будущих токенах в текущий шаг предсказания.

### Опора на уже подготовленный GPU-стек
- сначала используйте уже настроенное окружение из `00-Foundations`;
- не переустанавливайте тяжёлый стек «по кругу» без необходимости;
- в рамках ЛР не выполняются системные правки через `pkexec`.

Базовые материалы по локальному GPU-стеку:
- [../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md](../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md)
- [../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md](../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md)
- [../../00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md](../../00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md)

## Критерии завершения
### `ЛР01`
- тестовая токенная точность не ниже `0.97`;
- тестовая перплексия не выше `1.30`;
- корректное продолжение выполняется минимум в `18 из 20` фиксированных запусков;
- диагностика внимания подтверждает отсутствие доступа к будущим позициям.

### `ЛР02 CPU`
- `test_perplexity < baseline_perplexity`;
- `success_count >= cfg['gen_threshold']`;
- `mean_match_ratio >= cfg['gen_mean_threshold']`;
- диагностика внимания подтверждает отсутствие доступа в будущее.

### `ЛР02 GPU`
- `gpu_preflight()` пройден полностью;
- `test_perplexity < baseline_perplexity`;
- `test_perplexity < CPU_REFERENCE_PERPLEXITY` контролируется как индикатор;
- `success_count >= 19` (из `20`) в режиме контролируемого продолжения (teacher forcing);
- `mean_match_ratio >= cfg['gen_mean_threshold']` в режиме контролируемого продолжения;
- warm-up выполнен отдельно и не входит в измеряемый бюджет;
- остановка по generation-gate выполняется только после минимального времени timed-run и
  подтверждения цели на нескольких последовательных проверках;
- зафиксирована причина остановки: ранняя остановка, стабильное достижение генерационной цели или временного бюджета;
- свободная автогенерация выводится как демонстрационный блок, а не как проходной gate;
- диагностика внимания подтверждает отсутствие доступа в будущее.

## Запуск
Команды выполняются из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/04-Autoregression/lab/requirements.txt
python3 -m ipykernel install --user --name autoregression-lab --display-name "Python (.venv) Autoregression Lab"
.venv/bin/jupyter notebook
```

Для `ЛР02 GPU` дополнительно задайте режим запуска:

```bash
export COURSE_RUNTIME_MODE=local-gpu
```

Готовые скрипты запуска GPU-тетрадей:

```bash
themes/04-Autoregression/lab/scripts/setup_local_gpu_tf221.sh
themes/04-Autoregression/lab/scripts/run_gpu_starter.sh
themes/04-Autoregression/lab/scripts/run_gpu_solution.sh
themes/04-Autoregression/lab/scripts/execute_gpu_solution.sh
themes/04-Autoregression/lab/scripts/extract_gpu_run_summary.py
```

Назначение скриптов:
- `setup_local_gpu_tf221.sh` — подготавливает локальный GPU-стек для `RTX 50xx`: базовые зависимости ЛР04 + `tensorflow[and-cuda]==2.21.0` + `nvidia-cuda-nvcc-cu12>=12.8`, регистрирует Jupyter kernel.
- `run_gpu_starter.sh` — открывает стартовую GPU-тетрадь `02_decoder_only_tiny_shakespeare_gpu.ipynb` в Jupyter Notebook.
- `run_gpu_solution.sh` — открывает solution GPU-тетрадь `solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb` в Jupyter Notebook.
- `execute_gpu_solution.sh` — выполняет solution-тетрадь через `nbconvert` в неинтерактивном режиме в сценарии `warm-up + timed-run`, сохраняет выполненную копию даже при ошибках и формирует итоговую сводку `PASS/FAIL` по критерию `19/20`.
- `extract_gpu_run_summary.py` — извлекает строку `GPU_RUN_SUMMARY_JSON` из выполненной тетради и сохраняет отдельный `summary.json`, если лог запуска неполный.

Все три скрипта принудительно выставляют:
- `COURSE_RUNTIME_MODE=local-gpu`.

Recovery-блок для текущей `.venv` (выполнять только если `gpu_preflight()` не проходит):

```bash
python3 -m pip install --upgrade 'tensorflow[and-cuda]>=2.16,<2.20'
python3 -m pip install -r themes/04-Autoregression/lab/requirements.txt
```

Если `gpu_preflight()` уже проходит, повторную переустановку делать не нужно.

Переменные запуска для `execute_gpu_solution.sh`:
- `GPU_TRAINING_BUDGET_MINUTES` — измеряемый бюджет после warm-up (по умолчанию `60`).
- `GPU_WARMUP_STEPS` — число кратких warm-up train-итераций (по умолчанию `2`).
- `GPU_WARMUP_PROBE_STEPS` — длина warm-up generation-probe (по умолчанию `2`).
- `GPU_PROFILE_NAME` — профиль запуска (`gpu_60m` или `gpu_60m_boost`).

Если на видеокартах семейства `RTX 50xx` остаётся ошибка совместимости
(`CUDA_ERROR_INVALID_PTX`, `CUDA_ERROR_INVALID_HANDLE`, падение на XLA-autotuner),
выполните расширенный recovery:

```bash
python3 -m pip install --upgrade 'tensorflow[and-cuda]==2.21.0'
python3 -m pip install --upgrade 'nvidia-cuda-nvcc-cu12>=12.8'
```

После этого перезапустите ядро тетради и снова выполните `gpu_preflight()`.
В GPU-ноутбуках предусмотрена автоматическая предзагрузка CUDA-библиотек из `.venv`
до импорта TensorFlow.

## Как новичку проверить, что `ЛР02 GPU` пройдена
### Шаг 1. Запустите неинтерактивный сценарий
```bash
themes/04-Autoregression/lab/scripts/execute_gpu_solution.sh themes/04-Autoregression/lab/solutions/runs_stability_a
```

### Шаг 2. Найдите итоговые файлы
Ищите в каталоге запуска:
- `02_decoder_only_tiny_shakespeare_gpu_solution.summary.json`;
- при наличии `*.executed.ipynb` используйте его как дополнительный технический артефакт.

### Шаг 3. Примените минимальный чек PASS/FAIL
Считайте запуск успешным, если в `summary.json` одновременно:
- `overall_pass = true`;
- `success_count >= 19`;
- `generation_pass = true`;
- `baseline_pass = true`.

### Шаг 4. Корректно интерпретируйте `cpu_reference_pass`
Поле `cpu_reference_pass` в `ЛР02 GPU` индикаторное:
- `false` не означает автоматический провал;
- ключевой барьер прохождения — жёсткий критерий `19/20` и базовый критерий по перплексии.

### Шаг 5. Если лог короткий или оборван
При падении IDE (например, окно VS Code с `code: 139`) `run1.log` может остаться коротким,
но выполненная тетрадь при этом может быть корректно сохранена.

В таком случае восстановите `summary.json` из выполненной тетради:
```bash
python3 themes/04-Autoregression/lab/scripts/extract_gpu_run_summary.py \
  themes/04-Autoregression/lab/solutions/runs_stability_a/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb \
  --output themes/04-Autoregression/lab/solutions/runs_stability_a/02_decoder_only_tiny_shakespeare_gpu_solution.summary.json
```

### Канонический пример для ориентира
В репозитории зафиксирован учебный пример успешного прогона:
- [canonical_gpu_run_summary.json](./solutions/runs_stability_a/canonical_gpu_run_summary.json)
- [canonical_gpu_run_report.md](./solutions/runs_stability_a/canonical_gpu_run_report.md)

## Дополнительные материалы
- [../theory/theory.md](../theory/theory.md) — каноническая теория темы.
- [../../00-Foundations/showcases/cards/06_shakespeare.md](../../00-Foundations/showcases/cards/06_shakespeare.md) — карточка корпуса `Shakespeare`.
- [../../03-Transformer/theory/theory.md](../../03-Transformer/theory/theory.md) — предыдущий теоретический шаг.

## Что дальше
- [../../05-Full-Transformer/lab/README.md](../../05-Full-Transformer/lab/README.md) — финальная `ЛР05`: полный трансформер `encoder-decoder` (starter: `Tiny Shakespeare`, solution: `WikiText-2`).
