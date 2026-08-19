"""Экспорт чанков."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config_chunking import ChunkingPipelineConfig
from ..models_chunking import ChunkingExportResult, ChunkingValidationResult, PreparedChunk

LOGGER = logging.getLogger(__name__)


class ChunkExportStage:
    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config

    def run(self, chunks: list[PreparedChunk], validation_result: ChunkingValidationResult, run_id: str = "local-run") -> ChunkingExportResult:
        output_dir = Path(self.config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / self.config.paths.json_filename
        jsonl_path = output_dir / self.config.paths.jsonl_filename
        manifest_path = output_dir / self.config.paths.manifest_filename
        payload = [chunk.model_dump(mode="json") for chunk in chunks]
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in payload: f.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "config": self.config.model_dump(mode="json"), "counts": {"total_chunks": len(chunks), "empty_chunks": validation_result.empty_chunks_count, "undersized_chunks": validation_result.undersized_chunks_count, "oversized_chunks": validation_result.oversized_chunks_count, "low_quality_chunks": validation_result.low_quality_chunks_count}, "outputs": {"json": str(json_path), "jsonl": str(jsonl_path)}}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Экспортировано %d чанков в %s", len(chunks), output_dir)
        return ChunkingExportResult(json_path=json_path, jsonl_path=jsonl_path, manifest_path=manifest_path, chunks_count=len(chunks), run_id=run_id)