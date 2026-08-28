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

    input_jsonl = Path(cfg.paths.input_jsonl)

    # 1. Загрузка (теперь размерность из конфига, а не из файла)
    loader = EmbeddingsLoadingStage(expected_dimension=cfg.vector_store.vector_dimension)
    total_lines, valid_data, errors = loader.run(input_jsonl)

    # 2. Индексация
    indexer = QdrantIndexingStage(cfg.model_dump())
    indexer.setup_collection(vector_dimension=cfg.vector_store.vector_dimension)
    indexer.upload_embeddings(valid_data)

    # Получаем реальное количество точек в Qdrant после загрузки
    collection_info = indexer.client.get_collection(indexer.collection_name)
    actual_uploaded_count = collection_info.points_count or 0

    # 3. Валидация (передаем ожидаемую размерность)
    validator = VectorStoreValidationStage(
        indexer.client,
        indexer.collection_name,
        expected_dimension=cfg.vector_store.vector_dimension
    )
    validation_result = validator.run(expected_count=len(valid_data), actual_uploaded=actual_uploaded_count)

    # 4. Тестовый поиск (передаем seed)
    search_cfg = cfg.search
    searcher = TestSearchStage(
        indexer.client, indexer.collection_name, top_k=search_cfg.top_k, seed=cfg.run.seed
    )
    search_results = searcher.run(valid_data, num_queries=search_cfg.num_test_queries)

    # 5. Экспорт артефактов
    stats = {
        "total_input_lines": total_lines,
        "valid_embeddings": len(valid_data),
        "dropped_embeddings": total_lines - len(valid_data),
        "actual_uploaded_points": actual_uploaded_count
    }
    exporter = VectorStoreExportStage(cfg.model_dump())
    output_files = exporter.run(validation_result, search_results, stats, cfg.run.name)

    result = {
        "run_id": cfg.run.name,
        "input_count": total_lines,
        "valid_count": len(valid_data),
        "uploaded_count": actual_uploaded_count,
        "validation": validation_result,
        "search_tests": len(search_results),
        "output_files": {k: str(v) for k, v in output_files.items()}
    }

    LOGGER.info("Vector store pipeline завершён успешно")
    return result