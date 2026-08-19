"""Общие утилиты пайплайна."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    """Хеш файла для отслеживания изменений."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    """Хеш текста для дедупликации."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: object) -> str:
    """Детерминированный ID из набора частей."""
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def setup_logging(level: str = "INFO", log_file: str = "logs/pipeline.log") -> None:
    """Настраивает логирование."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )