"""Пайплайн чанкинга."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .chunking_stages import (
    PreparedDocumentLoadingStage,
    ChunkSplittingStage,
    ChunkValidationStage,
    ChunkExportStage,
)
from .config_chunking import ChunkingPipelineConfig
from .utils import setup_logging

LOGGER = logging.getLogger(__name__)


def run_chunking_pipeline(config: ChunkingPipelineConfig) -> dict[str, Any]:
    setup_logging(config.logging.level, "logs/chunking.log")
    stats: dict[str, Any] = {}

    loader = PreparedDocumentLoadingStage()
    documents = loader.run(Path(config.paths.input_jsonl))
    stats["loaded_documents"] = len(documents)

    splitter = ChunkSplittingStage(config.chunking)
    chunks = splitter.run(documents, run_id=config.run.name)
    stats["total_chunks"] = len(chunks)

    validator = ChunkValidationStage(config.chunking, config.validation)
    validation_result = validator.run(chunks)
    stats["validation"] = validation_result.model_dump()

    exporter = ChunkExportStage(config)
    export_result = exporter.run(chunks, validation_result, run_id=config.run.name)
    stats["export"] = export_result.model_dump(mode="json")

    LOGGER.info("Chunking pipeline completed: %s", stats)
    return {"stats": stats, "files": [str(export_result.json_path), str(export_result.jsonl_path), str(export_result.manifest_path)]}