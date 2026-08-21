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
        """Создает или пересоздает коллекцию."""
        distance_str = self.config["vector_store"].get("distance_metric", "Cosine")
        distance_enum = getattr(Distance, distance_str.upper(), Distance.COSINE)

        # Проверяем существование
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            exists = self.collection_name in collections
        except Exception:
            exists = False

        if exists and self.recreate:
            LOGGER.info("Удаление старой коллекции %s", self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            LOGGER.info(
                "Создание коллекции %s (dim=%d, metric=%s)",
                self.collection_name, vector_dimension, distance_str
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_dimension, distance=distance_enum)
            )
        else:
            LOGGER.info("Коллекция %s уже существует", self.collection_name)

    def upload_embeddings(self, data: list[dict[str, Any]]) -> int:
        """Загружает векторы батчами."""
        total_uploaded = 0

        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            points = []

            for record in batch:
                chunk_id = record["metadata"]["chunk_id"]

                # Генерируем детерминированный UUID из chunk_id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

                # Payload — это всё, что не вектор
                payload = {
                    "text": record["text"],
                    **record["metadata"]
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=record["embedding"],
                        payload=payload
                    )
                )

            # upsert = insert or update
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            total_uploaded += len(points)

            if total_uploaded % 500 == 0 or total_uploaded == len(data):
                LOGGER.info("Загружено %d / %d точек", total_uploaded, len(data))

        return total_uploaded