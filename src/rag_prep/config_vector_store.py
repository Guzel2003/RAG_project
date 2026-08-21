"""Конфигурация vector store."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str = "rag_vector_store"
    seed: int = 42


class PathsConfig(BaseModel):
    input_jsonl: str = "data/embeddings/embeddings.jsonl"
    output_dir: str = "data/vector_store"
    storage_dir: str = "data/qdrant_storage"


class VectorStoreConfig(BaseModel):
    provider: str = "qdrant_local"
    collection_name: str = "construction_docs"
    recreate_collection: bool = True
    distance_metric: str = "Cosine"
    batch_size: int = 100


class SearchConfig(BaseModel):
    top_k: int = 5
    num_test_queries: int = 3


class LoggingConfig(BaseModel):
    level: str = "INFO"


class VectorStorePipelineConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_vector_store_config(path: str | Path) -> VectorStorePipelineConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Vector store config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return VectorStorePipelineConfig.model_validate(data)