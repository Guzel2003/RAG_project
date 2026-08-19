"""Структурирование элементов в финальные документы."""

from __future__ import annotations

import logging
from collections import defaultdict

from ..config import StructuringConfig
from ..models import PreparedDocument, ProcessedElement

LOGGER = logging.getLogger(__name__)


class StructuringStage:
    """Группирует элементы в документы с metadata."""

    def __init__(self, config: StructuringConfig):
        self.config = config

    def run(self, elements: list[ProcessedElement]) -> list[PreparedDocument]:
        if self.config.group_by_section:
            groups: dict[tuple, list[ProcessedElement]] = defaultdict(list)
            for e in elements:
                key = (e.source_file.source_hash, tuple(e.section_path))
                groups[key].append(e)

            documents = []
            for (source_hash, section_path), group in groups.items():
                group.sort(key=lambda x: x.element_index)
                text = "\n\n".join(e.text for e in group)
                metadata = self._build_metadata(group[0], section_path, len(group))
                documents.append(PreparedDocument(text=text, metadata=metadata))
        else:
            # Группируем по файлу
            by_file: dict[str, list[ProcessedElement]] = defaultdict(list)
            for e in elements:
                by_file[e.source_file.source_hash].append(e)

            documents = []
            for source_hash, group in by_file.items():
                group.sort(key=lambda x: x.element_index)
                text = "\n\n".join(e.text for e in group)
                metadata = self._build_metadata(group[0], [self.config.default_section], len(group))
                documents.append(PreparedDocument(text=text, metadata=metadata))

        LOGGER.info("Structured %d elements into %d documents", len(elements), len(documents))
        return documents

    def _build_metadata(self, element: ProcessedElement, section_path: tuple, element_count: int) -> dict:
        meta = {
            "document_id": element.element_id,
            "source": element.source_file.source,
            "file_type": element.source_file.file_type,
            "section": " / ".join(section_path) if section_path else self.config.default_section,
            "source_hash": element.source_file.source_hash,
            "element_count": element_count,
            "lineage": {
                "origin_source": element.source_file.source,
                "stage": "prepared",
            },
        }
        meta.update({k: v for k, v in element.metadata.items() if k != "quality"})
        if "quality" in element.metadata:
            meta["quality"] = element.metadata["quality"]
        return meta