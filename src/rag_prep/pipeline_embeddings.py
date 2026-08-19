"""Пайплайн embeddings."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .embedding_stages import (
    ChunkLoadingStage,
    EmbeddingStage,
    EmbeddingValidationStage,
    MetricsCollectionStage,
    EmbeddingExportStage,
)
from .config_embeddings import load_embeddings_config
from .utils import setup_logging

LOGGER = logging.getLogger(__name__)


def run_embeddings_pipeline(config_path: str) -> dict[str, Any]:
    cfg = load_embeddings_config(config_path)
    setup_logging(cfg.logging.level, "logs/embeddings.log")

    LOGGER.info("Запуск embeddings с конфигом: %s", config_path)

    start_time = time.time()

    # 1. Загрузка
    input_jsonl = Path(cfg.paths.input_jsonl)
    loader = ChunkLoadingStage()
    chunks = loader.run(input_jsonl)
    LOGGER.info("Загружено чанков: %d", len(chunks))

    # 2. Расчёт embeddings
    embedder = EmbeddingStage(cfg.model_dump())
    embeddings = embedder.run(chunks, cfg.run.name)
    LOGGER.info("Получено embeddings: %d", len(embeddings))

    # 3. Валидация
    validator = EmbeddingValidationStage(cfg.model_dump())
    validation_result = validator.run(embeddings)

    # 4. Метрики
    processing_time = time.time() - start_time
    metrics_collector = MetricsCollectionStage()
    metrics = metrics_collector.run(embeddings, chunks, processing_time)

    # 5. Экспорт
    exporter = EmbeddingExportStage(cfg.model_dump())
    output_files = exporter.run(embeddings, validation_result, metrics, cfg.run.name)

    result = {
        "run_id": cfg.run.name,
        "chunks_count": len(chunks),
        "embeddings_count": len(embeddings),
        "validation": validation_result,
        "metrics": metrics,
        "output_files": {k: str(v) for k, v in output_files.items()},
    }

    LOGGER.info("Embeddings завершены успешно")
    return result