"""Экспорт артефактов проверки vector store."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class VectorStoreExportStage:
    """Сохраняет validation.json, search_results.json, manifest.json."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["paths"]["output_dir"])

    def run(
        self,
        validation_result: dict[str, Any],
        search_results: list[dict[str, Any]],
        stats: dict[str, Any],
        run_id: str = "local-run",
    ) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validation
        val_path = self.output_dir / "validation.json"
        val_path.write_text(json.dumps(validation_result, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Сохранён validation.json в %s", val_path)

        # Search Results
        search_path = self.output_dir / "search_results.json"
        search_path.write_text(json.dumps(search_results, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Сохранён search_results.json в %s", search_path)

        # Manifest
        manifest = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vector_store": self.config["vector_store"]["provider"],
            "collection_name": self.config["vector_store"]["collection_name"],
            "input_embeddings": stats.get("input_embeddings", 0),
            "uploaded_points": stats.get("uploaded_points", 0),
            "validation_status": validation_result.get("status"),
            "artifacts": {
                "validation": str(val_path),
                "search_results": str(search_path)
            }
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Сохранён manifest.json в %s", manifest_path)

        return {
            "validation": val_path,
            "search_results": search_path,
            "manifest": manifest_path
        }