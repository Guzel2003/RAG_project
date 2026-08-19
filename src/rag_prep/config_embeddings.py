"""Конфигурация пайплайна embeddings."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str = "rag_embeddings"
    seed: int = 42


class PathsConfig(BaseModel):
    input_jsonl: str = "data/chunks/chunks.jsonl"
    output_dir: str = "data/embeddings"
    json_filename: str = "embeddings.json"
    jsonl_filename: str = "embeddings.jsonl"
    manifest_filename: str = "manifest.json"


class EmbeddingConfig(BaseModel):
    provider: str = "local"
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 64
    max_tokens: int = 8191
    normalize: bool = True
    device: str = "cpu"


class RetryConfig(BaseModel):
    max_attempts: int = 3
    delay_seconds: int = 2


class ValidationConfig(BaseModel):
    fail_on_validation_error: bool = False
    check_dimensions: bool = True
    check_nan: bool = True
    check_duplicates: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"


class EmbeddingPipelineConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_embeddings_config(path: str | Path) -> EmbeddingPipelineConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return EmbeddingPipelineConfig.model_validate(data)