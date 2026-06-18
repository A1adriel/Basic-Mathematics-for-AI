# Диагностический порядок `ЛР05`

## 1. Проблема: не сходится обучение
Симптомы:
- `val_loss` не падает;
- `test_perplexity` близка к baseline.

Проверить:
1. Корректность формирования `decoder_input = [SOS] + target[:-1]`.
2. Согласованность длины `SRC_LEN`, `TGT_LEN` и окна данных.
3. Что seed фиксирован до построения выборок.

## 2. Проблема: генерация не проходит порог
Симптомы:
- низкий `success_count` и `mean_match_ratio`.

Проверить:
1. Что генерация выполняется жадно (`argmax`) без случайного сэмплирования.
2. Что prompts берутся из выбранного `probe_split` детерминированно.
3. Что во время генерации модель в режиме `training=False`.
4. Для starter: `mean_match_ratio` считается как точные `argmax`-совпадения по символам.
5. Для solution: `mean_match_ratio` считается как `top-k hit ratio`, а не как только `argmax`-совпадения.

## 3. Проблема: внимание «смотрит в будущее"
Симптомы:
- в верхнем треугольнике attention-карты заметные веса.

Проверить:
1. Что для decoder self-attention задан `use_causal_mask=True`.
2. Что padding mask и causal mask объединяются, а не подменяют друг друга.
3. Что диагностируете именно self-attention декодера, а не cross-attention.

## 4. Проблема: выбран `GPU-friendly`, но GPU не используется
Симптомы:
- обучение идёт долго как на CPU;
- `tf.config.list_physical_devices('GPU')` пуст.

Проверить:
1. Значение `COURSE_RUNTIME_PROFILE`.
2. Доступность устройства в TensorFlow.
3. Что тетрадь завершилась ранней ошибкой конфигурации, а не тихо перешла на CPU.
4. Что используется уже подготовленная `.venv` из темы `00-Foundations`, а не случайное окружение.
5. Что `gpu_preflight()` завершается статусом `PASSED` до начала длинного обучения.

Recovery-маршрут:
1. Свериться с [../../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md](../../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md).
2. При необходимости проверить версионную совместимость по [../../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md](../../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md).
3. Для карт семейства `RTX 50xx` использовать практический кейс из [../../../00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md](../../../00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md).

## 5. Мини-чек перед финальным выводом
- [ ] `test_perplexity < baseline_perplexity`.
- [ ] Порог генерации для выбранной тетради выполнен.
- [ ] Диагностика внимания подтвердила отсутствие утечки в будущее.
