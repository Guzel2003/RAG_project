"""Загрузка исходных файлов через LlamaIndex."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from ..config import LoaderConfig
from ..models import SourceFile
from ..utils import file_sha256

LOGGER = logging.getLogger(__name__)


class LlamaIndexLoadingStage:
    """Находит файлы через SimpleDirectoryReader."""

    def __init__(self, config: LoaderConfig):
        self.config = config

    def run(self, input_dir: Path) -> list[SourceFile]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")

        try:
            reader = SimpleDirectoryReader(
                input_dir=str(input_dir),
                recursive=self.config.recursive,
                required_exts=self.config.allowed_extensions,
                exclude_hidden=self.config.exclude_hidden,
                num_files_limit=self.config.num_files_limit,
                filename_as_id=True,
            )
            files = [Path(p) for p in reader.input_files]
        except ValueError as exc:
            if "No files found" not in str(exc):
                raise
            files = []

        sources = [self._to_source_file(p, input_dir) for p in sorted(files)]

        if not sources:
            raise ValueError(f"No supported files found in {input_dir}")

        LOGGER.info("Loaded %d source files from %s", len(sources), input_dir)
        return sources

    def _to_source_file(self, path: Path, input_dir: Path) -> SourceFile:
        stat = path.stat()
        resolved = path.resolve()
        try:
            source_key = resolved.relative_to(input_dir.resolve()).as_posix()
        except ValueError:
            source_key = path.name

        return SourceFile(
            path=resolved,
            source=str(resolved),
            source_key=source_key,
            file_name=path.name,
            file_type=path.suffix.lower().lstrip("."),
            source_hash=file_sha256(path),
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )