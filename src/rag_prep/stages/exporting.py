"""Экспорт подготовленных документов."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from ..models import PreparedDocument

LOGGER = logging.getLogger(__name__)


class ExportStage:
    """Сохраняет документы в JSON/JSONL + manifest."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, documents: list[PreparedDocument], stats: dict[str, Any]) -> list[str]:
        out_dir = Path(self.config.paths.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = [doc.model_dump(mode="json") for doc in documents]
        files: list[str] = []

        # JSONL
        jsonl_path = out_dir / self.config.paths.jsonl_filename
        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in payload:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        files.append(str(jsonl_path))

        # JSON
        json_path = out_dir / self.config.paths.json_filename
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        files.append(str(json_path))

        # Manifest
        manifest = {
            "run_id": self.config.run.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.model_dump(mode="json"),
            "stats": stats,
            "outputs": {"json": str(json_path), "jsonl": str(jsonl_path)},
        }
        manifest_path = out_dir / self.config.paths.manifest_filename
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        files.append(str(manifest_path))

        LOGGER.info("Exported %d documents to %s", len(documents), out_dir)
        return files