"""Загрузка чанков для embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class ChunkLoadingStage:
    """Загружает чанки из JSONL."""

    def run(self, input_jsonl: Path) -> list[dict]:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Файл чанков не существует: {input_jsonl}")

        chunks: list[dict] = []
        errors: list[str] = []

        with input_jsonl.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    if "text" not in chunk:
                        errors.append(f"Строка {line_num}: нет поля 'text'")
                        continue
                    if "metadata" not in chunk:
                        errors.append(f"Строка {line_num}: нет поля 'metadata'")
                        continue
                    if not chunk["text"].strip():
                        errors.append(f"Строка {line_num}: пустой текст")
                        continue
                    chunks.append(chunk)
                except json.JSONDecodeError as exc:
                    errors.append(f"Строка {line_num}: некорректный JSON: {exc}")

        if errors:
            LOGGER.warning("Найдено %d проблемных строк:\n%s", len(errors), "\n".join(errors[:10]))

        LOGGER.info("Загружено чанков: %d (ошибок: %d)", len(chunks), len(errors))

        if not chunks:
            raise ValueError("Не удалось загрузить ни одного чанка")

        return chunks