"""Расчёт embeddings через FastEmbed."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger(__name__)


class EmbeddingStage:
    """Рассчитывает embeddings для чанков."""

    def __init__(self, config: dict):
        self.config = config
        self.provider = config.get("embedding", {}).get("provider", "local")
        self.batch_size = int(config.get("embedding", {}).get("batch_size", 64))
        self.max_tokens = int(config.get("embedding", {}).get("max_tokens", 8191))
        self.normalize = bool(config.get("embedding", {}).get("normalize", True))
        self.model_name = config.get("embedding", {}).get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.retry_max = int(config.get("retry", {}).get("max_attempts", 3))
        self.retry_delay = int(config.get("retry", {}).get("delay_seconds", 2))

        self.model = None
        self.dimensions = 0
        self._init_model()

    def _init_model(self):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("fastembed не установлен. pip install fastembed")

        # FastEmbed использует короткие имена моделей
        # Маппинг из sentence-transformers в fastembed
        model_mapping = {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "intfloat/multilingual-e5-base": "intfloat/multilingual-e5-base",
        }

        fastembed_model = model_mapping.get(self.model_name, self.model_name)

        LOGGER.info("Загрузка модели FastEmbed: %s", fastembed_model)
        self.model = TextEmbedding(model_name=fastembed_model)

        # Определяем размерность
        test_emb = list(self.model.embed(["test"]))
        self.dimensions = len(test_emb[0])
        LOGGER.info("Модель загружена. Размерность: %d", self.dimensions)

    def run(self, chunks: list[dict], run_id: str = "local-run") -> list[dict[str, Any]]:
        start_time = time.time()
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_results = self._process_batch(batch, run_id, errors)
            results.extend(batch_results)

            progress = min(i + self.batch_size, len(chunks))
            if progress % (self.batch_size * 5) == 0 or progress == len(chunks):
                LOGGER.info("Прогресс: %d/%d чанков (%.1f%%)", progress, len(chunks), 100 * progress / len(chunks))

        elapsed = time.time() - start_time
        LOGGER.info("Embeddings рассчитаны: %d успешно, %d ошибок за %.2f сек", len(results), len(errors), elapsed)

        if errors:
            LOGGER.warning("Ошибки:\n%s", "\n".join(errors[:10]))

        return results

    def _process_batch(self, batch: list[dict], run_id: str, errors: list[str]) -> list[dict[str, Any]]:
        texts = [chunk["text"] for chunk in batch]

        # Проверка длины
        valid_indices = []
        for idx, text in enumerate(texts):
            estimated_tokens = len(text) / 4
            if estimated_tokens > self.max_tokens:
                chunk_id = batch[idx]["metadata"].get("chunk_id", "unknown")
                errors.append(f"Чанк {chunk_id}: слишком длинный ({int(estimated_tokens)} токенов)")
                continue
            valid_indices.append(idx)

        if not valid_indices:
            return []

        valid_texts = [texts[i] for i in valid_indices]

        # Retry механизм
        embeddings = None
        for attempt in range(self.retry_max):
            try:
                # FastEmbed возвращает генератор numpy массивов
                embeddings = list(self.model.embed(valid_texts, batch_size=self.batch_size))
                break
            except Exception as exc:
                if attempt < self.retry_max - 1:
                    LOGGER.warning("Попытка %d/%d не удалась: %s. Повтор через %d сек", attempt + 1, self.retry_max, exc, self.retry_delay)
                    time.sleep(self.retry_delay)
                else:
                    errors.append(f"Ошибка при обработке батча после {self.retry_max} попыток: {exc}")
                    return []

        if embeddings is None:
            return []

        # Нормализация
        if self.normalize:
            embeddings = self._normalize_embeddings(embeddings)

        # Формирование результатов
        results = []
        for idx, emb_idx in enumerate(valid_indices):
            chunk = batch[emb_idx]
            embedding = embeddings[idx].tolist() if hasattr(embeddings[idx], "tolist") else list(embeddings[idx])
            result = self._create_embedding_record(chunk, embedding, run_id)
            results.append(result)

        return results

    def _normalize_embeddings(self, embeddings) -> list[list[float]]:
        try:
            import numpy as np
            arr = np.array(embeddings)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return (arr / norms).tolist()
        except ImportError:
            normalized = []
            for emb in embeddings:
                norm = math.sqrt(sum(x * x for x in emb))
                if norm > 0:
                    normalized.append([x / norm for x in emb])
                else:
                    normalized.append(list(emb))
            return normalized

    def _create_embedding_record(self, chunk: dict, embedding: list[float], run_id: str) -> dict[str, Any]:
        chunk_meta = chunk.get("metadata", {})
        text_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        embedding_hash = hashlib.sha256(str(embedding[:10]).encode("utf-8")).hexdigest()

        metadata = {
            "chunk_id": chunk_meta.get("chunk_id", chunk_meta.get("id", "unknown")),
            "document_id": chunk_meta.get("document_id", "unknown"),
            "embedding_model": self.model_name,
            "embedding_provider": self.provider,
            "embedding_dimensions": self.dimensions,
            "embedding_run_id": run_id,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
            "text_hash": text_hash,
            "embedding_hash": embedding_hash,
            "lineage": {
                "origin_source": chunk_meta.get("source"),
                "origin_document_id": chunk_meta.get("document_id"),
                "embedding_model_version": self.model_name,
            },
            "source": chunk_meta.get("source"),
            "section": chunk_meta.get("section"),
            "position": chunk_meta.get("position"),
        }

        return {
            "text": chunk["text"],
            "embedding": embedding,
            "metadata": metadata,
        }