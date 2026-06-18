# Входной минимум перед `ЛР05`

## Что студент уже должен уметь
1. Работать с тензорами ранга `2` и `3` в TensorFlow.
2. Понимать маскирование в самовнимании (self-attention) и смысл причинной маски (causal mask).
3. Интерпретировать метрики `loss`, `accuracy`, `perplexity`.

## Требование к окружению для GPU
1. Для режима `GPU-friendly` сначала используйте уже подготовленную `.venv` из `00-Foundations`.
2. Не переустанавливайте полный стек без необходимости: сначала проверьте, что TensorFlow видит `GPU`.
3. Если `GPU` не виден, переходите к recovery-порядку из guide `05` темы `00-Foundations`.
4. В `ЛР05` обязательный первый барьер для `GPU-friendly` — успешный `gpu_preflight(): PASSED`.

Полезные ссылки:
- [../../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md](../../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md)
- [../../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md](../../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md)
- [../../../00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md](../../../00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md)

## Что новое именно в `ЛР05`
1. Полная архитектура «кодировщик-декодировщик» (encoder-decoder).
2. Перекрёстное внимание (cross-attention) между декодером и выходами кодировщика.
3. Контракт `decoder_input/decoder_target` для teacher forcing.

## Что важно не перепутать
1. `encoder_input` и `decoder_input` всегда разные по смыслу.
2. Причинная маска относится к self-attention декодера, а не к encoder self-attention.
3. Оценка качества делается и по перплексии, и по детерминированной генерации.

## Мини-чек перед стартом
- [ ] Я понимаю формулу цепного правила вероятностей.
- [ ] Я понимаю, зачем нужен `SOS` в `decoder_input`.
- [ ] Я понимаю, как читать `success_count` и `mean_match_ratio` в контролируемой генерации.
