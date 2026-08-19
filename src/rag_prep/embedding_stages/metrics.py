"""Сбор метрик качества embeddings."""

from __future__ import annotations

import logging
import math
from typing import Any

LOGGER = logging.getLogger(__name__)


class MetricsCollectionStage:
    """Собирает статистику пайплайна."""

    def run(self, embeddings: list[dict[str, Any]], chunks: list[dict], processing_time: float) -> dict[str, Any]:
        if not embeddings:
            return {
                "total_chunks": len(chunks),
                "successful_embeddings": 0,
                "failed_embeddings": len(chunks),
                "avg_text_length": 0.0,
                "avg_embedding_norm": 0.0,
                "processing_time_seconds": processing_time,
                "chunks_per_second": 0.0,
            }

        # Средняя длина текста
        text_lengths = [len(e["text"]) for e in embeddings]
        avg_text_length = sum(text_lengths) / len(text_lengths)

        # Средняя норма вектора
        norms = []
        for e in embeddings:
            emb = e["embedding"]
            norm = math.sqrt(sum(x * x for x in emb))
            norms.append(norm)
        avg_norm = sum(norms) / len(norms) if norms else 0.0

        first_meta = embeddings[0]["metadata"]

        metrics = {
            "total_chunks": len(chunks),
            "successful_embeddings": len(embeddings),
            "failed_embeddings": len(chunks) - len(embeddings),
            "avg_text_length": round(avg_text_length, 2),
            "avg_embedding_norm": round(avg_norm, 4),
            "embedding_model": first_meta.get("embedding_model"),
            "embedding_dimensions": first_meta.get("embedding_dimensions"),
            "processing_time_seconds": round(processing_time, 2),
            "chunks_per_second": round(len(embeddings) / processing_time, 2) if processing_time > 0 else 0.0,
        }

        LOGGER.info(
            "Метрики: %d embeddings, средняя длина текста: %.0f символов, средняя норма: %.4f, скорость: %.2f chunks/sec",
            len(embeddings),
            avg_text_length,
            avg_norm,
            metrics["chunks_per_second"],
        )

        return metrics