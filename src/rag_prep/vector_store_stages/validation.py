"""Проверка состояния Vector Store после загрузки."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class VectorStoreValidationStage:
    """Проверяет, что количество точек и payload корректны."""

    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def run(self, expected_count: int) -> dict[str, Any]:
        info = self.client.get_collection(self.collection_name)

        actual_count = info.points_count or 0

        # В новых версиях qdrant-client vectors_count может отсутствовать
        # Используем points_count как fallback
        vectors_count = getattr(info, 'vectors_count', actual_count) or actual_count

        has_points = actual_count > 0
        payload_ok = False
        sample_point = None

        if has_points:
            # Достаем 1 случайную точку для проверки payload
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            if points:
                sample_point = points[0]
                payload = sample_point.payload
                required_fields = ["text", "chunk_id", "document_id"]
                payload_ok = all(field in payload for field in required_fields)

        result = {
            "status": "success" if (actual_count == expected_count and payload_ok) else "failed",
            "expected_points": expected_count,
            "actual_points": actual_count,
            "actual_vectors": vectors_count,
            "counts_match": actual_count == expected_count,
            "payload_structure_ok": payload_ok,
            "sample_chunk_id": sample_point.payload.get("chunk_id") if sample_point else None
        }

        LOGGER.info("Результат валидации БД: %s", result["status"])
        return result