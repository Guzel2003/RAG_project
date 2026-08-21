"""Тестовый поиск по векторной базе."""

from __future__ import annotations

import logging
import random
from typing import Any

LOGGER = logging.getLogger(__name__)


class TestSearchStage:
    """Выполняет тестовые запросы для проверки работы retrieval."""

    def __init__(self, client, collection_name: str, top_k: int = 5):
        self.client = client
        self.collection_name = collection_name
        self.top_k = top_k

    def run(self, source_data: list[dict[str, Any]], num_queries: int) -> list[dict[str, Any]]:
        # Выбираем случайные чанки для теста
        test_samples = random.sample(source_data, min(num_queries, len(source_data)))

        results = []

        for sample in test_samples:
            query_vector = sample["embedding"]
            query_text = sample["text"][:100] + "..."

            # Ищем в Qdrant (новый API)
            try:
                # Новая версия qdrant-client использует query_points
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=self.top_k,
                    with_payload=True
                ).points
            except AttributeError:
                # Fallback для старых версий
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=self.top_k,
                    with_payload=True
                )

            formatted_hits = []
            for hit in search_result:
                formatted_hits.append({
                    "score": round(hit.score, 4),
                    "chunk_id": hit.payload.get("chunk_id"),
                    "document_id": hit.payload.get("document_id"),
                    "text_preview": hit.payload.get("text", "")[:100] + "..."
                })

            # Проверяем релевантность: первый результат должен быть тем же самым чанком
            is_relevant = False
            if formatted_hits:
                is_relevant = (formatted_hits[0]["chunk_id"] == sample["metadata"]["chunk_id"])

            results.append({
                "query_text_preview": query_text,
                "query_chunk_id": sample["metadata"]["chunk_id"],
                "top_k_results": formatted_hits,
                "is_first_result_relevant": is_relevant
            })

        LOGGER.info("Выполнено %d тестовых запросов", len(results))
        return results