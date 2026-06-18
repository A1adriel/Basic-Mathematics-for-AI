# Канонический отчёт для новичка: `ЛР02 GPU` пройдена

Этот отчёт фиксирует успешный контрольный запуск `ЛР02 GPU` в теме `04`.
Главный приёмочный комплект для студента:
- `canonical_gpu_run_summary.json`;
- этот поясняющий отчёт.

Выполненная тетрадь `*.executed.ipynb` может храниться как технический артефакт,
но для учебной проверки достаточно пары `MD+JSON`.

## Где лежат артефакты
- JSON: `themes/04-Autoregression/lab/solutions/runs_stability_a/canonical_gpu_run_summary.json`
- Отчёт: `themes/04-Autoregression/lab/solutions/runs_stability_a/canonical_gpu_run_report.md`

## Ключевые значения
- `overall_pass = true`
- `generation_pass = true`
- `baseline_pass = true`
- `success_count = 20` из `20`
- `mean_match_ratio = 0.415625`
- `test_perplexity = 8.686638837916096`
- `baseline_perplexity = 26.929200704052754`
- `stop_reason = early_stopping_val_plateau`
- `timed_elapsed_minutes = 45.389097386983245`

## Как это читать простыми словами
1. Жёсткий порог `19/20` выполнен с запасом: получено `20/20`.
2. Средняя доля совпадений (`mean_match_ratio`) выше порога, значит модель
   устойчиво продолжает последовательность в контролируемом режиме.
3. Перплексия (perplexity) на тесте лучше частотного ориентира
   (`8.6866 < 26.9292`), то есть модель учится закономерностям корпуса.
4. Остановка `early_stopping_val_plateau` означает, что полезный прирост по
   валидационной функции потерь стабилизировался. Это штатная остановка.
5. Значение `cpu_reference_pass = false` не ломает прохождение, потому что в
   `ЛР02 GPU` это индикаторный контроль, а не аварийный барьер.

## Мини-чек для студента
Считайте запуск пройденным, если одновременно выполнено:
- `overall_pass = true`;
- `success_count >= 19`;
- `generation_pass = true`;
- `baseline_pass = true`.

Если IDE (например, VS Code) упала и лог короткий, ориентируйтесь на
`summary.json`: он является источником итогового статуса.
