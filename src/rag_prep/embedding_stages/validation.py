"""Валидация embeddings."""

from __future__ import annotations

import logging
import math
from typing import Any

LOGGER = logging.getLogger(__name__)


class EmbeddingValidationStage:
    """Проверяет качество embeddings."""

    def __init__(self, config: dict):
        self.config = config
        self.check_dimensions = bool(config.get("validation", {}).get("check_dimensions", True))
        self.check_nan = bool(config.get("validation", {}).get("check_nan", True))
        self.check_duplicates = bool(config.get("validation", {}).get("check_duplicates", True))
        self.fail_on_error = bool(config.get("validation", {}).get("fail_on_validation_error", False))

    def run(self, embeddings: list[dict[str, Any]]) -> dict[str, int]:
        result = {
            "total_embeddings": len(embeddings),
            "empty_embeddings_count": 0,
            "wrong_dimension_count": 0,
            "nan_count": 0,
            "infinity_count": 0,
            "duplicate_ids_count": 0,
        }

        if not embeddings:
            result["empty_embeddings_count"] = 1
            return result

        # Проверка размерности
        if self.check_dimensions:
            expected_dim = len(embeddings[0]["embedding"])
            for emb in embeddings:
                if len(emb["embedding"]) != expected_dim:
                    result["wrong_dimension_count"] += 1

        # Проверка NaN и Infinity
        if self.check_nan:
            for emb in embeddings:
                for value in emb["embedding"]:
                    if math.isnan(value):
                        result["nan_count"] += 1
                        break
                    if math.isinf(value):
                        result["infinity_count"] += 1
                        break

        # Проверка дубликатов
        if self.check_duplicates:
            ids = [e["metadata"]["chunk_id"] for e in embeddings]
            unique_ids = set(ids)
            result["duplicate_ids_count"] = len(ids) - len(unique_ids)

        LOGGER.info(
            "Валидация embeddings: всего=%d, пустые=%d, неверная размерность=%d, NaN=%d, Infinity=%d, дубликаты=%d",
            result["total_embeddings"],
            result["empty_embeddings_count"],
            result["wrong_dimension_count"],
            result["nan_count"],
            result["infinity_count"],
            result["duplicate_ids_count"],
        )

        has_errors = (
            result["empty_embeddings_count"] > 0
            or result["wrong_dimension_count"] > 0
            or result["nan_count"] > 0
            or result["infinity_count"] > 0
        )

        if has_errors and self.fail_on_error:
            raise ValueError(f"Валидация embeddings завершилась ошибкой: {result}")

        return result