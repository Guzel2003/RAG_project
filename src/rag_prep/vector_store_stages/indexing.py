"""Создание коллекции Qdrant и загрузка векторов."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

LOGGER = logging.getLogger(__name__)


class QdrantIndexingStage:
    """Управляет коллекцией Qdrant и загрузкой векторов."""

    def __init__(self, config: dict):
        self.config = config
        self.storage_dir = Path(config["paths"]["storage_dir"])
        self.collection_name = config["vector_store"]["collection_name"]
        self.recreate = config["vector_store"].get("recreate_collection", True)
        self.batch_size = config["vector_store"].get("batch_size", 100)

        # Инициализируем локальный клиент Qdrant
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.storage_dir))

    def setup_collection(self, vector_dimension: int):
        distance_str = self.config["vector_store"]["distance_metric"]

        # Пункт 7: Fail-fast для метрики
        distance_str_upper = distance_str.upper()
        if distance_str_upper not in Distance.__members__:
            raise ValueError(
                f"Недопустимая метрика расстояния в конфиге: '{distance_str}'. Допустимые: {list(Distance.__members__.keys())}")

        distance_enum = Distance[distance_str_upper]

        try:
            collections = [c.name for c in self.client.get_collections().collections]
            exists = self.collection_name in collections
        except Exception:
            exists = False

        if exists:
            if self.recreate:
                LOGGER.info("Удаление старой коллекции %s", self.collection_name)
                self.client.delete_collection(self.collection_name)
                exists = False
            else:
                # Пункт 6: сверка параметров существующей коллекции
                collection_info = self.client.get_collection(self.collection_name)
                actual_dim = collection_info.config.params.vectors.size
                actual_dist = collection_info.config.params.vectors.distance

                if actual_dim != vector_dimension:
                    raise ValueError(
                        f"Несоответствие размерности: в конфиге {vector_dimension}, в существующей коллекции {actual_dim}")
                if actual_dist != distance_enum:
                    raise ValueError(
                        f"Несоответствие метрики: в конфиге {distance_enum}, в существующей коллекции {actual_dist}")

                LOGGER.info("Коллекция %s существует и её параметры совпадают с конфигурацией", self.collection_name)
                return

        LOGGER.info("Создание коллекции %s (dim=%d, metric=%s)", self.collection_name, vector_dimension, distance_str)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dimension, distance=distance_enum)
        )

    def upload_embeddings(self, data: list[dict[str, Any]]) -> int:
        unique_point_ids = set()  # Пункт 8: считаем уникальные точки, а не попытки

        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            points = []

            for record in batch:
                chunk_id = record["metadata"]["chunk_id"]
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
                unique_point_ids.add(point_id)

                payload = {"text": record["text"], **record["metadata"]}
                points.append(PointStruct(id=point_id, vector=record["embedding"], payload=payload))

            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            LOGGER.info("Загружено уникальных точек: %d / %d записей", len(unique_point_ids), len(data))

        return len(unique_point_ids)