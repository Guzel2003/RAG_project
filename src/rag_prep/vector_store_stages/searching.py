"""Тестовый поиск по векторной базе."""

from __future__ import annotations

import logging
import random
from typing import Any

LOGGER = logging.getLogger(__name__)


class TestSearchStage:
    """Выполняет тестовые запросы для проверки работы retrieval."""

    def __init__(self, client, collection_name: str, top_k: int = 5, seed: int = 42):
        self.client = client
        self.collection_name = collection_name
        self.top_k = top_k
        self.seed = seed

    def run(self, source_data: list[dict[str, Any]], num_queries: int) -> list[dict[str, Any]]:
        random.seed(self.seed)  # Пункт 12: фиксируем seed для воспроизводимости
        test_samples = random.sample(source_data, min(num_queries, len(source_data)))

        results = []
        for sample in test_samples:
            query_vector = sample["embedding"]
            query_text_full = sample["text"]  # Пункт 13: сохраняем полный текст запроса

            try:
                search_result = self.client.query_points(
                    collection_name=self.collection_name, query=query_vector, limit=self.top_k, with_payload=True
                ).points
            except AttributeError:
                search_result = self.client.search(
                    collection_name=self.collection_name, query_vector=query_vector, limit=self.top_k, with_payload=True
                )

            formatted_hits = []
            for hit in search_result:
                # Пункт 13: сохраняем полную metadata, а не только превью
                formatted_hits.append({
                    "score": round(hit.score, 4),
                    "chunk_id": hit.payload.get("chunk_id"),
                    "document_id": hit.payload.get("document_id"),
                    "text": hit.payload.get("text", ""),
                    "metadata": {k: v for k, v in hit.payload.items() if k not in ["text", "chunk_id", "document_id"]}
                })

            # Пункт 14: уточняем логику релевантности
            is_exact_match = False
            is_in_top_k = False
            if formatted_hits:
                is_exact_match = (formatted_hits[0]["chunk_id"] == sample["metadata"]["chunk_id"])
                is_in_top_k = any(h["chunk_id"] == sample["metadata"]["chunk_id"] for h in formatted_hits)

            results.append({
                "query_text": query_text_full,
                "query_chunk_id": sample["metadata"]["chunk_id"],
                "top_k_results": formatted_hits,
                "is_exact_match_first": is_exact_match,
                "is_relevant_in_top_k": is_in_top_k
            })

        LOGGER.info("Выполнено %d тестовых запросов", len(results))
        return results