"""Оркестрация этапов пайплайна."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .stages import (
    LlamaIndexLoadingStage,
    UnstructuredParsingStage,
    TextCleaningStage,
    TextNormalizationStage,
    DeduplicationStage,
    StructuringStage,
    ExportStage,
)
from .utils import setup_logging

LOGGER = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    setup_logging(config.logging.level)
    stats: dict[str, Any] = {}

    # 1. Loading
    loader = LlamaIndexLoadingStage(config.loader)
    sources = loader.run(Path(config.paths.input_dir))
    stats["source_files"] = len(sources)

    # 2. Parsing
    parser = UnstructuredParsingStage(config.parser, config.structuring.default_section)
    parse_result = parser.run(sources)
    stats["parsed_elements"] = len(parse_result.elements)
    stats["parse_failures"] = len(parse_result.failures)

    # 3. Cleaning
    cleaner = TextCleaningStage(config.cleaning)
    cleaned = cleaner.run(parse_result.elements)
    stats["cleaned_elements"] = len(cleaned)

    # 4. Normalization
    normalizer = TextNormalizationStage(config.normalization)
    normalized = normalizer.run(cleaned)
    stats["normalized_elements"] = len(normalized)

    # 5. Deduplication
    deduplicator = DeduplicationStage(config.deduplication)
    dedup_result = deduplicator.run(normalized)
    stats["deduplicated_elements"] = len(dedup_result.elements)
    stats["duplicates_removed"] = dedup_result.duplicates_removed

    # 6. Structuring
    structurer = StructuringStage(config.structuring)
    documents = structurer.run(dedup_result.elements)
    stats["prepared_documents"] = len(documents)

    # 7. Export
    exporter = ExportStage(config)
    output_files = exporter.run(documents, stats)
    stats["output_files"] = output_files

    LOGGER.info("Pipeline completed: %s", stats)
    return {"stats": stats, "files": output_files}