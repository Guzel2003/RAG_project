"""Конфигурация чанкинга."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str = "rag_chunking"
    seed: int = 42


class PathsConfig(BaseModel):
    input_jsonl: str = "data/prepared/documents.jsonl"
    output_dir: str = "data/chunks"
    json_filename: str = "chunks.json"
    jsonl_filename: str = "chunks.jsonl"
    manifest_filename: str = "manifest.json"


class ChunkingConfig(BaseModel):
    strategy: str = "sentence"
    chunk_size: int = 512
    chunk_overlap: int = 50
    tokenizer_model: str = "cl100k_base"
    embedding_model: str = "text-embedding-3-small"
    preserve_section_boundaries: bool = True
    preserve_block_boundaries: bool = True
    min_chunk_tokens: int = 50
    max_chunk_tokens: int = 1024
    min_quality_score: float = 0.3


class ValidationConfig(BaseModel):
    fail_on_validation_error: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"


class ChunkingPipelineConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_chunking_config(path: str | Path) -> ChunkingPipelineConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chunking config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return ChunkingPipelineConfig.model_validate(data)