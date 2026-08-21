"""Загрузка embeddings для vector store."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class EmbeddingsLoadingStage:
    """Загружает и валидирует embeddings из JSONL."""

    def __init__(self, expected_dimension: int | None = None):
        self.expected_dimension = expected_dimension

    def run(self, input_jsonl: Path) -> list[dict[str, Any]]:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Файл embeddings не найден: {input_jsonl}")

        data: list[dict[str, Any]] = []
        errors: list[str] = []

        with input_jsonl.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Строка {line_num}: битый JSON ({e})")
                    continue

                # Проверка структуры
                if "text" not in record or "embedding" not in record or "metadata" not in record:
                    errors.append(f"Строка {line_num}: нет text, embedding или metadata")
                    continue

                chunk_id = record["metadata"].get("chunk_id")
                if not chunk_id:
                    errors.append(f"Строка {line_num}: нет chunk_id в metadata")
                    continue

                # Проверка вектора
                vector = record["embedding"]
                if not isinstance(vector, list) or len(vector) == 0:
                    errors.append(f"Строка {line_num}: пустой или некорректный вектор")
                    continue

                has_bad_values = any(math.isnan(v) or math.isinf(v) for v in vector)
                if has_bad_values:
                    errors.append(f"Строка {line_num}: в векторе есть NaN или Infinity")
                    continue

                # Проверка размерности
                if self.expected_dimension and len(vector) != self.expected_dimension:
                    errors.append(
                        f"Строка {line_num}: неверная размерность "
                        f"(ожидалось {self.expected_dimension}, получили {len(vector)})"
                    )
                    continue

                data.append(record)

        if errors:
            LOGGER.warning("Найдено %d ошибок валидации. Примеры:\n%s", len(errors), "\n".join(errors[:5]))

        LOGGER.info("Успешно загружено и проверено %d embeddings", len(data))

        if not data:
            raise ValueError("Нет валидных данных для загрузки в Vector Store")

        return data