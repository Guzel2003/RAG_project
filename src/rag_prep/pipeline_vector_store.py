"""Пайплайн vector store."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .vector_store_stages import (
    EmbeddingsLoadingStage,
    QdrantIndexingStage,
    VectorStoreValidationStage,
    TestSearchStage,
    VectorStoreExportStage,
)
from .config_vector_store import load_vector_store_config
from .utils import setup_logging

LOGGER = logging.getLogger(__name__)


def run_vector_store_pipeline(config_path: str) -> dict[str, Any]:
    cfg = load_vector_store_config(config_path)
    setup_logging(cfg.logging.level, "logs/vector_store.log")

    LOGGER.info("Запуск vector store с конфигом: %s", config_path)

    # Определяем размерность из первой записи
    input_jsonl = Path(cfg.paths.input_jsonl)
    with input_jsonl.open("r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        dimension = len(first_line["embedding"])

    # 1. Загрузка и проверка данных
    loader = EmbeddingsLoadingStage(expected_dimension=dimension)
    data = loader.run(input_jsonl)
    LOGGER.info("Загружено embeddings: %d", len(data))

    # 2. Индексация
    indexer = QdrantIndexingStage(cfg.model_dump())
    indexer.setup_collection(vector_dimension=dimension)
    uploaded_count = indexer.upload_embeddings(data)
    LOGGER.info("Загружено точек: %d", uploaded_count)

    # 3. Валидация базы
    validator = VectorStoreValidationStage(indexer.client, indexer.collection_name)
    validation_result = validator.run(expected_count=len(data))

    # 4. Тестовый поиск
    search_cfg = cfg.search
    searcher = TestSearchStage(
        indexer.client,
        indexer.collection_name,
        top_k=search_cfg.top_k
    )
    search_results = searcher.run(data, num_queries=search_cfg.num_test_queries)

    # 5. Экспорт артефактов
    stats = {
        "input_embeddings": len(data),
        "uploaded_points": uploaded_count
    }
    exporter = VectorStoreExportStage(cfg.model_dump())
    output_files = exporter.run(validation_result, search_results, stats, cfg.run.name)

    result = {
        "run_id": cfg.run.name,
        "input_count": len(data),
        "uploaded_count": uploaded_count,
        "validation": validation_result,
        "search_tests": len(search_results),
        "output_files": {k: str(v) for k, v in output_files.items()}
    }

    LOGGER.info("Vector store pipeline завершён успешно")
    return result