"""Модели данных для vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VectorStoreValidationResult(BaseModel):
    """Результат валидации индекса."""
    status: str = "unknown"
    expected_points: int = 0
    actual_points: int = 0
    actual_vectors: int = 0
    counts_match: bool = False
    payload_structure_ok: bool = False
    sample_chunk_id: str | None = None


class SearchResult(BaseModel):
    """Результат одного поискового запроса."""
    query_text_preview: str
    query_chunk_id: str
    top_k_results: list[dict[str, Any]] = Field(default_factory=list)
    is_first_result_relevant: bool = False


class VectorStoreExportResult(BaseModel):
    """Результат экспорта артефактов."""
    validation_path: Path
    search_results_path: Path
    manifest_path: Path
    run_id: str