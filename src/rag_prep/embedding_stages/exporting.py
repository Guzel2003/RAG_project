"""Экспорт embeddings."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class EmbeddingExportStage:
    """Сохраняет embeddings в JSON/JSONL + manifest."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get("paths", {}).get("output_dir", "data/embeddings"))
        self.json_filename = config.get("paths", {}).get("json_filename", "embeddings.json")
        self.jsonl_filename = config.get("paths", {}).get("jsonl_filename", "embeddings.jsonl")
        self.manifest_filename = config.get("paths", {}).get("manifest_filename", "manifest.json")

    def run(
        self,
        embeddings: list[dict[str, Any]],
        validation_result: dict[str, int],
        metrics: dict[str, Any],
        run_id: str = "local-run",
    ) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / self.json_filename
        jsonl_path = self.output_dir / self.jsonl_filename
        manifest_path = self.output_dir / self.manifest_filename

        # JSON
        json_path.write_text(json.dumps(embeddings, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Сохранён embeddings.json в %s", json_path)

        # JSONL
        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in embeddings:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        LOGGER.info("Сохранён embeddings.jsonl в %s", jsonl_path)

        # Manifest
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "provider": self.config.get("embedding", {}).get("provider"),
                "model": self.config.get("embedding", {}).get("model"),
                "batch_size": self.config.get("embedding", {}).get("batch_size"),
                "normalize": self.config.get("embedding", {}).get("normalize"),
            },
            "counts": {
                "total_embeddings": len(embeddings),
                "empty_embeddings": validation_result.get("empty_embeddings_count", 0),
                "wrong_dimension": validation_result.get("wrong_dimension_count", 0),
                "nan_count": validation_result.get("nan_count", 0),
                "duplicate_ids": validation_result.get("duplicate_ids_count", 0),
            },
            "metrics": metrics,
            "outputs": {
                "json": str(json_path),
                "jsonl": str(jsonl_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Сохранён manifest.json в %s", manifest_path)

        return {"json": json_path, "jsonl": jsonl_path, "manifest": manifest_path}