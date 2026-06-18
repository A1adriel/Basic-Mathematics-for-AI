#!/usr/bin/env python3
"""Извлекает итог `GPU_RUN_SUMMARY_JSON` из выполненной тетради.

Скрипт нужен для ситуации, когда лог запуска прервался, но выполненная
тетрадь (`*.executed.ipynb`) сохранилась на диске.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def extract_text_chunks(output_obj: dict) -> list[str]:
    """Собирает текстовые фрагменты из output-объекта ячейки.

    Args:
      output_obj: Словарь одного output-блока из формата `ipynb`.

    Returns:
      Список строковых фрагментов, в которых может находиться итоговый JSON.

    Raises:
      RuntimeError: Не выбрасывается в штатном режиме.
    """
    chunks: list[str] = []
    text_value = output_obj.get("text")
    if isinstance(text_value, list):
        chunks.extend(str(item) for item in text_value)
    elif text_value is not None:
        chunks.append(str(text_value))

    data_obj = output_obj.get("data", {})
    plain_value = data_obj.get("text/plain")
    if isinstance(plain_value, list):
        chunks.extend(str(item) for item in plain_value)
    elif plain_value is not None:
        chunks.append(str(plain_value))
    return chunks


def iter_code_outputs(notebook: dict) -> Iterable[dict]:
    """Итерирует output-блоки только кодовых ячеек.

    Args:
      notebook: Загруженный JSON тетради.

    Returns:
      Итератор output-объектов кодовых ячеек.

    Raises:
      RuntimeError: Не выбрасывается в штатном режиме.
    """
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            yield output


def extract_gpu_run_summary(notebook_path: Path) -> dict:
    """Извлекает словарь сводки запуска из выполненной тетради.

    Args:
      notebook_path: Путь к `*.executed.ipynb`.

    Returns:
      Распарсенный словарь из строки `GPU_RUN_SUMMARY_JSON=...`.

    Raises:
      FileNotFoundError: Если файл тетради не найден.
      RuntimeError: Если в тетради нет строки `GPU_RUN_SUMMARY_JSON=...`.
      json.JSONDecodeError: Если найденный JSON повреждён.
    """
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Не найден файл тетради: {notebook_path}")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for output in iter_code_outputs(notebook):
        for chunk in extract_text_chunks(output):
            for line in chunk.splitlines():
                if line.startswith("GPU_RUN_SUMMARY_JSON="):
                    payload = line.split("=", 1)[1].strip()
                    return json.loads(payload)

    raise RuntimeError(
        "В выполненной тетради не найдена строка GPU_RUN_SUMMARY_JSON=..."
    )


def resolve_output_path(notebook_path: Path, output_path: Path | None) -> Path:
    """Возвращает путь для сохранения итогового `summary.json`.

    Args:
      notebook_path: Путь к выполненной тетради.
      output_path: Явно заданный путь сохранения или `None`.

    Returns:
      Путь итогового JSON-файла.

    Raises:
      RuntimeError: Не выбрасывается в штатном режиме.
    """
    if output_path is not None:
        return output_path

    name = notebook_path.name
    marker = ".executed.ipynb"
    if marker in name:
        return notebook_path.with_name(name.replace(marker, ".summary.json"))
    return notebook_path.with_suffix(".summary.json")


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Args:
      Нет аргументов на вход функции.

    Returns:
      Пространство имён с полями `notebook` и `output`.

    Raises:
      SystemExit: При ошибке разбора аргументов.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Извлечь GPU_RUN_SUMMARY_JSON из выполненной тетради и сохранить "
            "в отдельный summary.json."
        )
    )
    parser.add_argument(
        "notebook",
        type=Path,
        help="Путь к выполненной тетради (*.executed.ipynb).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Куда сохранить summary.json. Если не указан, путь вычисляется "
            "автоматически рядом с тетрадью."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Запускает извлечение и сохранение итоговой сводки.

    Args:
      Нет аргументов на вход функции.

    Returns:
      Код завершения процесса (`0` при успешном извлечении).

    Raises:
      SystemExit: При ошибке извлечения или записи файла.
    """
    args = parse_args()
    summary = extract_gpu_run_summary(args.notebook)
    destination = resolve_output_path(args.notebook, args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] Summary extracted to: {destination}")
    print(
        "[INFO] Ключевые поля: "
        f"overall_pass={summary.get('overall_pass')}, "
        f"success_count={summary.get('success_count')}/"
        f"{summary.get('prompt_count')}, "
        f"mean_match_ratio={summary.get('mean_match_ratio')}, "
        f"test_perplexity={summary.get('test_perplexity')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
