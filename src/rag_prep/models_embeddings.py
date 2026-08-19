"""Модели данных для embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EmbeddingMetadata(BaseModel):
    """Метаданные embedding вектора."""
    chunk_id: str
    document_id: str
    embedding_model: str
    embedding_provider: str
    embedding_dimensions: int
    embedding_run_id: str
    embedded_at: str
    text_hash: str
    embedding_hash: str | None = None
    lineage: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    section: str | None = None
    position: int | None = None


class EmbeddingRecord(BaseModel):
    """Запись с текстом, embedding и метаданными."""
    text: str
    embedding: list[float]
    metadata: EmbeddingMetadata


class EmbeddingValidationResult(BaseModel):
    """Результат валидации embeddings."""
    total_embeddings: int = 0
    empty_embeddings_count: int = 0
    wrong_dimension_count: int = 0
    nan_count: int = 0
    infinity_count: int = 0
    duplicate_ids_count: int = 0

    @property
    def has_errors(self) -> bool:
        return (
            self.empty_embeddings_count > 0
            or self.wrong_dimension_count > 0
            or self.nan_count > 0
            or self.infinity_count > 0
        )


class EmbeddingMetrics(BaseModel):
    """Метрики качества embeddings."""
    total_chunks: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    avg_text_length: float = 0.0
    avg_embedding_norm: float = 0.0
    embedding_model: str = ""
    embedding_dimensions: int = 0
    processing_time_seconds: float = 0.0
    chunks_per_second: float = 0.0


class EmbeddingExportResult(BaseModel):
    """Результат экспорта embeddings."""
    json_path: Path
    jsonl_path: Path
    manifest_path: Path
    embeddings_count: int
    run_id: str