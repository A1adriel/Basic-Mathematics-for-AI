# Лабораторный практикум `05-Full-Transformer`

## Назначение
`ЛР05` — финальная лабораторная курса: полноценный трансформер «кодировщик-декодировщик» (encoder-decoder Transformer) для предсказания токенов текста.

В теме используются два корпуса:
1. `starter`: `Tiny Shakespeare`.
2. `solution`: `WikiText-2` (текстовые `train/valid/test` из GitHub-источника).

## Структура
- `01_full_transformer_tiny_shakespeare.ipynb` — стартовая тетрадь (`TODO 1..6`).
- `solutions/01_full_transformer_wikitext2_solution.ipynb` — полное решение.
- `guides/00_full_transformer_prerequisites.md` — входной минимум.
- `guides/01_full_transformer_walkthrough.md` — пошаговый разбор.
- `guides/02_full_transformer_debugging_playbook.md` — диагностика.
- `../theory/theory.md` — теоретическая основа темы.
- `requirements.txt` — зависимости.

## Канонический маршрут студента
1. Прочитать [../theory/theory.md](../theory/theory.md).
2. Прочитать [guides/00_full_transformer_prerequisites.md](./guides/00_full_transformer_prerequisites.md).
3. Пройти [guides/01_full_transformer_walkthrough.md](./guides/01_full_transformer_walkthrough.md).
4. Выполнить [01_full_transformer_tiny_shakespeare.ipynb](./01_full_transformer_tiny_shakespeare.ipynb): для `GPU-friendly` сначала пройти `gpu_preflight()`.
5. При затруднениях использовать [guides/02_full_transformer_debugging_playbook.md](./guides/02_full_transformer_debugging_playbook.md).
6. Свериться с [solutions/01_full_transformer_wikitext2_solution.ipynb](./solutions/01_full_transformer_wikitext2_solution.ipynb).

## Обязательный workflow внутри ноутбуков
`контракт данных -> интуиция -> формализация -> ручной пример -> TODO -> mini-checks -> диагностика`.

## Контракт данных
Для каждого окна:
- `encoder_input = ids[i : i + SRC_LEN]`;
- `target = ids[i + SRC_LEN : i + SRC_LEN + TGT_LEN]`;
- `decoder_input = [SOS] + target[:-1]`;
- `decoder_target = target`.

## Контракт масок
- `padding mask`: запрещает учитывать `PAD`-позиции в encoder и decoder.
- `causal mask`: запрещает decoder self-attention смотреть в будущие токены.
- `cross-attention mask`: ограничивает доступ decoder только к валидным позициям `encoder_input`.

Минимальный контроль перед обучением:
1. Проверить, что `encoder_input`, `decoder_input`, `decoder_target` имеют ожидаемые длины.
2. Проверить, что `causal mask` нижнетреугольная.
3. Проверить, что `cross-attention` не включает `PAD`-позиции источника.

## Профили выполнения
- `CPU-friendly`: целевой бюджет `40-60` минут.
- `GPU-friendly`: целевой бюджет `30-45` минут.

Если выбран `GPU-friendly`, но `GPU` недоступен, ноутбук завершает выполнение ранней диагностической ошибкой без скрытого fallback.

## Контракт окружения для `GPU-friendly`
Для `GPU`-запуска сначала переиспользуйте уже подготовленное окружение из темы `00-Foundations`.
Повторная «тяжёлая» установка TensorFlow с CUDA не является обязательным шагом, если `GPU` уже корректно виден в текущей `.venv`.

Опорные материалы:
- [../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md](../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md)
- [../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md](../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md)
- [../../00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md](../../00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md)

Выбор профиля:
```bash
export COURSE_RUNTIME_PROFILE=CPU-friendly
# или
export COURSE_RUNTIME_PROFILE=GPU-friendly
```

## Критерии завершения
### Starter (`Tiny Shakespeare`)
- `test_perplexity < baseline_perplexity`;
- `success_count >= 18` из `20`;
- `mean_match_ratio >= 0.70` (в starter это доля точных `argmax`-совпадений по символам);
- для `GPU-friendly`: `gpu_preflight()` пройден полностью;
- диагностика внимания подтверждает отсутствие доступа к будущим позициям.

### Solution (`WikiText-2`)
- `test_perplexity < baseline_perplexity`;
- `success_count >= 16` из `20`;
- `mean_match_ratio >= 0.60` (для solution считается как `top-k hit ratio` в детерминированной генерации);
- для `GPU-friendly`: `gpu_preflight()` пройден полностью;
- диагностика внимания подтверждает отсутствие доступа к будущим позициям.

## Запуск
### Базовый сценарий (чистый старт)
Команды из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/05-Full-Transformer/lab/requirements.txt
python3 -m ipykernel install --user --name full-transformer-lab --display-name "Python (.venv) Full Transformer Lab"
.venv/bin/jupyter notebook
```

### Рекомендуемый сценарий для уже настроенного `GPU`
Если `.venv` уже подготовлена в теме `00-Foundations` и `GPU` виден в TensorFlow:
1. Активируйте существующую `.venv`.
2. Не переустанавливайте весь стек заново.
3. Выполните только лёгкую синхронизацию зависимостей темы `05`, если нужно.

```bash
source .venv/bin/activate
python3 -m pip install -r themes/05-Full-Transformer/lab/requirements.txt
```

Recovery (только если `GPU-friendly` выбран, но TensorFlow не видит `GPU`):
```bash
python3 -m pip install --upgrade 'tensorflow[and-cuda]>=2.16,<2.22'
python3 -m pip install -r themes/05-Full-Transformer/lab/requirements.txt
```

В рамках `ЛР05` не выполняются системные правки уровня ОС (`pkexec`, переустановка драйверов).

## Переходы по курсу
- Предыдущая тема: [../../04-Autoregression/lab/README.md](../../04-Autoregression/lab/README.md)
