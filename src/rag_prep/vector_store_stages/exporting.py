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
        self.artifacts = config["paths"].get("artifacts", {
            "validation": "validation.json",
            "search_results": "search_results.json",
            "manifest": "manifest.json"
        })

    def run(self, validation_result: dict[str, Any], search_results: list[dict[str, Any]], stats: dict[str, Any],
            run_id: str = "local-run") -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        val_path = self.output_dir / self.artifacts["validation"]
        val_path.write_text(json.dumps(validation_result, ensure_ascii=False, indent=2), encoding="utf-8")

        search_path = self.output_dir / self.artifacts["search_results"]
        search_path.write_text(json.dumps(search_results, ensure_ascii=False, indent=2), encoding="utf-8")

        # Пункт 15: добавляем конфиг и детали валидации в манифест
        manifest = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_snapshot": self.config,
            "stats": stats,
            "validation_status": validation_result.get("status"),
            "validation_details": validation_result,
            "artifacts": {
                "validation": str(val_path),
                "search_results": str(search_path),
                "manifest": str(self.output_dir / self.artifacts["manifest"])
            }
        }
        manifest_path = self.output_dir / self.artifacts["manifest"]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Сохранены артефакты в %s", self.output_dir)

        return {"validation": val_path, "search_results": search_path, "manifest": manifest_path}