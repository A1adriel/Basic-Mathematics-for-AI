# Лабораторный практикум по теме «Трансформер-кодировщик»

## Назначение
Практикум включает две последовательные лабораторные работы:
1. `Transformer encoder` на синтетической задаче, чувствительной к порядку токенов.
2. `Transformer encoder` на реальном корпусе `IMDB` для бинарной классификации тональности.

Цель блока — последовательно перевести студента от перекрёстного внимания (cross-attention) к самовниманию (self-attention), позиционным векторам (positional embedding) и архитектуре только с кодировщиком (encoder-only).

## Структура
- `01_transformer_encoder_order_toy.ipynb` — стартовая тетрадь с `TODO`.
- `02_transformer_encoder_imdb.ipynb` — стартовая тетрадь с `TODO`.
- `solutions/01_transformer_encoder_order_toy_solution.ipynb` — полное решение `ЛР01`.
- `solutions/02_transformer_encoder_imdb_solution.ipynb` — полное решение `ЛР02`.

Решения предназначены для сравнения после двух самостоятельных попыток.

## Карта курса
Общая линия курса:
1. Шаг 1 = `01-RNN / ЛР01`
2. Шаг 2 = `01-RNN / ЛР02`
3. Шаг 3 = `01-RNN / ЛР03`
4. Шаг 4 = `02-Attention / ЛР01`
5. Шаг 5 = `03-Transformer / ЛР01`
6. Шаг 6 = `03-Transformer / ЛР02`

Важно: этот блок намеренно ограничен вариантом `encoder-only`. Декодерный режим и причинная маска (causal mask) вынесены в следующий блок: [../../04-Autoregression/lab/README.md](../../04-Autoregression/lab/README.md).

## Канонический маршрут студента
1. Прочитать [guides/00_transformer_prerequisites.md](./guides/00_transformer_prerequisites.md).
2. Прочитать [guides/01_self_attention_and_positional_encoding_beginner.md](./guides/01_self_attention_and_positional_encoding_beginner.md).
3. Пройти [guides/02_transformer_encoder_toy_walkthrough.md](./guides/02_transformer_encoder_toy_walkthrough.md).
4. Выполнить [01_transformer_encoder_order_toy.ipynb](./01_transformer_encoder_order_toy.ipynb).
5. При затруднениях использовать [guides/04_transformer_debugging_playbook.md](./guides/04_transformer_debugging_playbook.md) и сделать вторую попытку.
6. Сравнить с [solutions/01_transformer_encoder_order_toy_solution.ipynb](./solutions/01_transformer_encoder_order_toy_solution.ipynb).
7. Пройти [guides/03_transformer_encoder_imdb_walkthrough.md](./guides/03_transformer_encoder_imdb_walkthrough.md).
8. Выполнить [02_transformer_encoder_imdb.ipynb](./02_transformer_encoder_imdb.ipynb), переиспользуя готовые блоки из `ЛР01`.
9. При затруднениях снова использовать диагностический гайд и выполнить вторую попытку.
10. Сравнить с [solutions/02_transformer_encoder_imdb_solution.ipynb](./solutions/02_transformer_encoder_imdb_solution.ipynb).

## Методическая траектория
Траектория в обеих работах одинакова:
1. Интуиция задачи.
2. Формализация (тензоры, маски, формулы).
3. Контроль форм и промежуточных проверок.
4. Диагностика внимания.
5. Итоговые метрики.

Бытовая аналогия: сначала мы проверяем, что карта маршрута читаема и согласована с местностью, и только потом оцениваем скорость движения. В лабораторных роль такой «карты» выполняют формы тензоров и маски.

## Beginner-мостик: encoder, self-attention и positional
- `encoder` в этой теме — это блок, который перечитывает всю последовательность токенов и обновляет представление каждого токена с учетом остальных.
- `self-attention` можно читать как таблицу «какой токен на какой токен смотрит» внутри одной и той же последовательности.
- `positional` часть нужна, чтобы отличать порядок токенов: без позиционных признаков модель видит «мешок токенов», а не последовательность.

Перед первым запуском держите три вопроса:
1. Где в текущей ячейке формируется `padding_mask`?
2. Где получают `attention_scores` для диагностики?
3. Какой участок кода отвечает за `positional` информацию?

## Критерии завершения
### `03-Transformer / ЛР01`
Работа считается завершённой, если одновременно выполнено:
- `test_acc >= 0.95`;
- два ручных примера с перестановкой `7` и `3` дают различающиеся предсказания;
- карта внимания строится на непустом фрагменте, без хвоста `PAD`.

### `03-Transformer / ЛР02`
Работа считается завершённой, если одновременно выполнено:
- `test_acc >= 0.75` на текущем учебном подмножестве;
- показан хотя бы один декодированный отзыв;
- карта внимания построена по содержательному фрагменту, а не по дополнению `PAD`.

## Формы тензоров
Базовые обозначения:
- `N` — число объектов;
- `T` — длина последовательности;
- `E` — размер пространства признаков;
- `H` — число голов внимания;
- `V` — размер словаря.

Ключевые формы:
- `tokens`: `(N, T)`
- `padding_mask`: `(N, T)`
- `embeddings`: `(N, T, E)`
- `attention_scores`: `(N, H, T, T)`
- `y_pred`: `(N, 1)`

## Запуск
Команды выполняются из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/03-Transformer/lab/requirements.txt
python3 -m ipykernel install --user --name transformer-lab --display-name "Python (.venv) Transformer Lab"
.venv/bin/jupyter notebook
```

### Варианты среды выполнения
- локально: `auto`, `local-cpu`, `local-gpu`;
- в `Google Colab`: `colab-cpu`, `colab-gpu`;
- в `Kaggle`: `kaggle-cpu`, `kaggle-gpu`.

Подробные инструкции:
- [../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md](../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md)
- [../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md](../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md)

## Связь с теорией
- [../theory/theory.md](../theory/theory.md) — каноническая теория блока.
- [../../02-Attention/theory/theory.md](../../02-Attention/theory/theory.md) — теоретическая база предыдущего шага.

## Что дальше
Следующий шаг:
- [../../04-Autoregression/lab/README.md](../../04-Autoregression/lab/README.md) — декодерный трансформер (decoder-only Transformer) и причинная маска.
