"""Валидация чанков."""

from __future__ import annotations

import logging

from ..config_chunking import ChunkingConfig, ValidationConfig
from ..models_chunking import ChunkingValidationResult, PreparedChunk

LOGGER = logging.getLogger(__name__)


class ChunkValidationStage:
    def __init__(self, chunking_config: ChunkingConfig, validation_config: ValidationConfig):
        self.chunking_config = chunking_config
        self.validation_config = validation_config

    def run(self, chunks: list[PreparedChunk]) -> ChunkingValidationResult:
        result = ChunkingValidationResult(
            no_chunks_count=int(not chunks),
            empty_chunks_count=sum(1 for c in chunks if not c.text.strip()),
            undersized_chunks_count=sum(1 for c in chunks if c.metadata.chunk_token_count < self.chunking_config.min_chunk_tokens),
            oversized_chunks_count=sum(1 for c in chunks if c.metadata.chunk_token_count > self.chunking_config.max_chunk_tokens),
            estimated_offsets_count=sum(1 for c in chunks if "estimated" in c.metadata.offset_strategy),
            missing_parent_count=sum(1 for c in chunks if not c.metadata.parent_ids),
            missing_lineage_count=sum(1 for c in chunks if not c.metadata.lineage),
            low_quality_chunks_count=sum(1 for c in chunks if c.metadata.quality.get("is_low_quality_chunk")),
        )
        LOGGER.info("Валидация: всего=%d empty=%d undersized=%d oversized=%d low_quality=%d", len(chunks), result.empty_chunks_count, result.undersized_chunks_count, result.oversized_chunks_count, result.low_quality_chunks_count)
        if result.no_chunks_count: raise ValueError("Чанкинг не сформировал ни одного чанка")
        if self.validation_config.fail_on_validation_error and result.has_errors: raise ValueError(f"Валидация завершилась ошибкой: {result.model_dump()}")
        return result