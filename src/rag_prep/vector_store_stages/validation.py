"""Проверка состояния Vector Store после загрузки."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class VectorStoreValidationStage:
    """Проверяет, что количество точек и payload корректны."""

    def __init__(self, client, collection_name: str, expected_dimension: int):
        self.client = client
        self.collection_name = collection_name
        self.expected_dimension = expected_dimension

    def run(self, expected_count: int, actual_uploaded: int) -> dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        actual_points_count = info.points_count or 0

        actual_dimension = info.config.params.vectors.size
        actual_distance = str(info.config.params.vectors.distance)
        dimension_match = (actual_dimension == self.expected_dimension)

        # Пункт 10: проверяем не 1, а до 10 точек
        sample_size = min(10, actual_points_count)
        payload_checks = {"text": 0, "chunk_id": 0, "document_id": 0, "metadata_ok": 0}

        if sample_size > 0:
            points, _ = self.client.scroll(
                collection_name=self.collection_name, limit=sample_size, with_payload=True, with_vectors=False
            )
            for pt in points:
                payload = pt.payload or {}
                if "text" in payload and isinstance(payload["text"], str) and payload["text"].strip():
                    payload_checks["text"] += 1
                if "chunk_id" in payload:
                    payload_checks["chunk_id"] += 1
                if "document_id" in payload:
                    payload_checks["document_id"] += 1
                if len(payload) > 1:
                    payload_checks["metadata_ok"] += 1

        # Пункт 11: детализированный результат
        result = {
            "status": "success" if (actual_points_count == expected_count and dimension_match and payload_checks[
                "chunk_id"] == sample_size) else "failed",
            "expected_points": expected_count,
            "actual_points": actual_points_count,
            "actual_uploaded_from_pipeline": actual_uploaded,
            "counts_match": actual_points_count == expected_count,
            "expected_dimension": self.expected_dimension,
            "actual_dimension": actual_dimension,
            "dimension_match": dimension_match,
            "actual_distance_metric": actual_distance,
            "payload_structure_checks": {
                "text_present_in_sample": f"{payload_checks['text']}/{sample_size}",
                "chunk_id_present_in_sample": f"{payload_checks['chunk_id']}/{sample_size}",
                "document_id_present_in_sample": f"{payload_checks['document_id']}/{sample_size}",
                "metadata_rich_in_sample": f"{payload_checks['metadata_ok']}/{sample_size}"
            },
            "sample_chunk_id": points[0].payload.get("chunk_id") if points else None
        }
        LOGGER.info("Результат валидации БД: %s", result["status"])
        return result