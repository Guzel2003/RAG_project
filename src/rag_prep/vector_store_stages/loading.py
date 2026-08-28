"""Загрузка embeddings для vector store."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class EmbeddingsLoadingStage:
    def __init__(self, expected_dimension: int):
        self.expected_dimension = expected_dimension

    def run(self, input_jsonl: Path) -> tuple[int, list[dict[str, Any]], list[str]]:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Файл embeddings не найден: {input_jsonl}")

        valid_data: list[dict[str, Any]] = []
        errors: list[str] = []
        total_lines = 0

        with input_jsonl.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                total_lines += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Строка {line_num}: битый JSON ({e})")
                    continue

                # 1. Проверка text (Пункт 3)
                if not isinstance(record.get("text"), str) or not record["text"].strip():
                    errors.append(f"Строка {line_num}: поле text отсутствует, не строка или пусто")
                    continue

                # 2. Проверка metadata (Пункт 3)
                if not isinstance(record.get("metadata"), dict):
                    errors.append(f"Строка {line_num}: metadata отсутствует или не является объектом")
                    continue

                meta = record["metadata"]
                if not meta.get("chunk_id") or not meta.get("document_id") or \
                        not meta.get("embedding_model") or not meta.get("embedding_dimensions"):
                    errors.append(
                        f"Строка {line_num}: в metadata отсутствуют обязательные поля (chunk_id, document_id, embedding_model, embedding_dimensions)")
                    continue

                # 3. Проверка вектора (Пункт 4)
                vector = record.get("embedding")
                if not isinstance(vector, list) or len(vector) == 0:
                    errors.append(f"Строка {line_num}: embedding отсутствует или не является непустым списком")
                    continue

                if len(vector) != self.expected_dimension:
                    errors.append(
                        f"Строка {line_num}: неверная размерность (ожидалось {self.expected_dimension}, получено {len(vector)})")
                    continue

                # 4. Проверка числовых значений (Пункт 4)
                has_bad_values = False
                for i, v in enumerate(vector):
                    if not isinstance(v, (int, float)):
                        errors.append(
                            f"Строка {line_num}: элемент вектора [{i}] не является числом (тип: {type(v).__name__})")
                        has_bad_values = True
                        break
                    if math.isnan(v) or math.isinf(v):
                        errors.append(f"Строка {line_num}: элемент вектора [{i}] содержит NaN или Infinity")
                        has_bad_values = True
                        break

                if has_bad_values:
                    continue

                valid_data.append(record)

        if errors:
            LOGGER.warning("Найдено %d ошибок валидации. Первые 5:\n%s", len(errors), "\n".join(errors[:5]))

        LOGGER.info("Обработано строк: %d, успешно: %d, отброшено: %d",
                    total_lines, len(valid_data), total_lines - len(valid_data))

        if not valid_data:
            raise ValueError("Нет валидных данных для загрузки в Vector Store")

        # Возвращаем кортеж (Пункт 5: чтобы пайплайн знал реальное количество входных строк)
        return total_lines, valid_data, errors