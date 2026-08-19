"""Модели данных для чанкинга."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    id: str
    document_id: str
    source: str
    section: str
    position: int
    chunk_start_char: int
    chunk_end_char: int
    chunk_token_count: int
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    tokenizer_model: str
    embedding_model: str
    chunking_run_id: str
    semantic_block_ids: list[str] = Field(default_factory=list)
    semantic_block_start: int | None = None
    semantic_block_end: int | None = None
    offset_strategy: str
    parent_ids: list[str] = Field(default_factory=list)
    origin_element_ids: list[str] = Field(default_factory=list)
    lineage: dict[str, Any] = Field(default_factory=dict)
    hierarchy: dict[str, Any] = Field(default_factory=dict)
    source_hash: str
    document_text_hash: str
    text_hash: str
    file_name: str
    file_type: str
    quality: dict[str, Any] = Field(default_factory=dict)


class PreparedChunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class ChunkingValidationResult(BaseModel):
    no_chunks_count: int = 0
    empty_chunks_count: int = 0
    undersized_chunks_count: int = 0
    oversized_chunks_count: int = 0
    estimated_offsets_count: int = 0
    missing_parent_count: int = 0
    missing_lineage_count: int = 0
    low_quality_chunks_count: int = 0

    @property
    def has_errors(self) -> bool:
        return self.no_chunks_count > 0 or self.empty_chunks_count > 0 or self.oversized_chunks_count > 0


class ChunkingExportResult(BaseModel):
    json_path: Path
    jsonl_path: Path
    manifest_path: Path
    chunks_count: int
    run_id: str