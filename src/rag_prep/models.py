"""Контракты данных для пайплайна подготовки."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SourceFile(BaseModel):
    """Метаданные исходного файла."""
    path: Path
    source: str
    source_key: str
    file_name: str
    file_type: str
    source_hash: str
    size_bytes: int
    modified_at: datetime


class RawElement(BaseModel):
    """Распарсенный элемент документа."""
    source_file: SourceFile
    element_id: str
    element_index: int
    text: str
    element_type: str
    section: str
    section_path: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedElement(BaseModel):
    """Элемент после очистки и нормализации."""
    source_file: SourceFile
    element_id: str
    element_index: int
    text: str
    element_type: str
    section: str
    section_path: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseFailure(BaseModel):
    """Информация об ошибке парсинга."""
    source: str
    file_name: str
    file_type: str
    error_type: str
    error_message: str


class ParseResult(BaseModel):
    """Результат этапа парсинга."""
    elements: list[RawElement]
    failures: list[ParseFailure]


class PreparedDocument(BaseModel):
    """Финальный документ для экспорта."""
    text: str
    metadata: dict[str, Any]