"""Загрузка подготовленных документов."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..models import PreparedDocument

LOGGER = logging.getLogger(__name__)


class PreparedDocumentLoadingStage:
    def run(self, input_jsonl: Path) -> list[PreparedDocument]:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Файл не существует: {input_jsonl}")
        documents: list[PreparedDocument] = []
        with input_jsonl.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip(): continue
                try: documents.append(PreparedDocument.model_validate(json.loads(line)))
                except Exception as exc: raise ValueError(f"Некорректный документ в {input_jsonl}:{line_num}") from exc
        LOGGER.info("Загружено документов: %d из %s", len(documents), input_jsonl)
        return documents