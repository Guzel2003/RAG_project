"""Загрузка и валидация конфигурации."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str = "rag_data_preparation"
    seed: int = 42


class PathsConfig(BaseModel):
    input_dir: str = "data/raw"
    output_dir: str = "data/prepared"
    json_filename: str = "documents.json"
    jsonl_filename: str = "documents.jsonl"
    manifest_filename: str = "manifest.json"


class LoaderConfig(BaseModel):
    recursive: bool = True
    allowed_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".txt", ".html", ".htm", ".csv", ".docx"])
    exclude_hidden: bool = True
    num_files_limit: int | None = None


class ParserConfig(BaseModel):
    strategy: str = "fast"
    encoding: str = "utf-8"
    fail_on_error: bool = False
    languages: list[str] = Field(default_factory=lambda: ["rus", "eng"])
    pdf_infer_table_structure: bool = False
    skip_infer_table_types: list[str] = Field(default_factory=lambda: ["pdf", "jpg", "png", "heic"])
    docx_include_tables: bool = True
    docx_table_separator: str = " | "


class CleaningConfig(BaseModel):
    min_chars: int = 12
    normalize_whitespace: bool = True
    remove_control_chars: bool = True
    drop_patterns: list[str] = Field(default_factory=list)
    boilerplate_patterns: list[str] = Field(default_factory=lambda: [
        "cookie|cookies|куки|cookie policy",
        "подписаться|subscribe|newsletter",
        "навигация|главное меню|menu",
        "©|copyright|all rights reserved",
    ])


class NormalizationConfig(BaseModel):
    unicode_form: str = "NFKC"
    lowercase: bool = False
    spacy_language: str = "ru"
    collect_sentence_stats: bool = True


class DeduplicationConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.9
    num_perm: int = 128
    shingle_size: int = 5
    min_tokens: int = 8


class StructuringConfig(BaseModel):
    group_by_section: bool = True
    default_section: str = "full_document"


class LoggingConfig(BaseModel):
    level: str = "INFO"


class PipelineConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    structuring: StructuringConfig = Field(default_factory=StructuringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> PipelineConfig:
    """Загружает YAML-конфиг и возвращает валидированную Pydantic-модель."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(data)