"""Структурирование элементов в финальные документы."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from ..config import StructuringConfig
from ..models import PreparedDocument, ProcessedElement

LOGGER = logging.getLogger(__name__)


def extract_document_metadata(source_path: str) -> dict[str, str]:
    """
    Извлекает метаданные документа (имя, категорию, версию)
    на основе имени исходного файла.
    Вызывается один раз для каждого уникального файла.
    """
    filename = Path(source_path).stem.lower()

    # Метаданные по умолчанию
    meta = {
        "document_name": Path(source_path).name,
        "category": "unknown",
        "version": "unknown"
    }

    # Маппинг ключевых слов в именах файлов -> нормализованные метаданные
    if any(k in filename for k in ["градостроительный кодекс", "грк"]):
        meta.update({
            "document_name": "Градостроительный кодекс РФ",
            "category": "legislation",
            "version": "2004-12-29"
        })
    elif "51872" in filename:
        meta.update({
            "document_name": "ГОСТ Р 51872-2019 Исполнительная геодезическая документация",
            "category": "standard",
            "version": "2019"
        })
    elif any(k in filename for k in ["рд-11-02-2006", "рд1102"]):
        meta.update({
            "document_name": "РД-11-02-2006 Требования к составу и порядку ведения исполнительной документации",
            "category": "guideline",
            "version": "2006"
        })
    elif any(k in filename for k in ["сп 48", "сп48", "снип 12-01", "организация строительства"]):
        meta.update({
            "document_name": "СП 48.13330.2019 Организация строительства",
            "category": "code_of_practice",
            "version": "2022-03-28"
        })
    elif "1004" in filename and "пр" in filename:
        meta.update({
            "document_name": "Приказ Минстроя России от 28.12.2023 N 1004/пр",
            "category": "order",
            "version": "2023-12-28"
        })
    elif "344" in filename and "пр" in filename:
        if any(k in filename for k in ["app3", "аоср", "скрытых"]):
            doc_name = "Приказ Минстроя №344-пр Приложение 3: Акт освидетельствования скрытых работ"
        elif any(k in filename for k in ["app4", "критических", "ответственных"]):
            doc_name = "Приказ Минстроя №344-пр Приложение 4: Акт освидетельствования ответственных конструкций"
        else:
            doc_name = "Приказ Минстроя №344-пр: Состав и порядок ведения исполнительной документации"
        meta.update({
            "document_name": doc_name,
            "category": "order",
            "version": "2023-05-16"
        })
    elif any(k in filename for k in ["sp-70", "70.13330"]):
        meta.update({
            "document_name": "СП 70.13330.2012 Несущие и ограждающие конструкции",
            "category": "code_of_practice",
            "version": "2012"
        })
    elif any(k in filename for k in ["84480d8e", "21.101"]):
        meta.update({
            "document_name": "ГОСТ Р 21.101-2020 Система проектной документации для строительства",
            "category": "standard",
            "version": "2020"
        })

    return meta


class StructuringStage:
    """Группирует элементы в документы с metadata."""

    def __init__(self, config: StructuringConfig):
        self.config = config
        # Кэш метаданных, чтобы не парсить имя файла для каждого элемента
        self._metadata_cache: dict[str, dict] = {}

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

    def _get_file_metadata(self, element: ProcessedElement) -> dict:
        """Получает или вычисляет метаданные файла с использованием кэша."""
        source = element.source_file.source
        if source not in self._metadata_cache:
            self._metadata_cache[source] = extract_document_metadata(source)
        return self._metadata_cache[source]

    def _build_metadata(self, element: ProcessedElement, section_path: tuple, element_count: int) -> dict:
        # 👇 ПОЛУЧАЕМ НАШИ НОВЫЕ МЕТАДАННЫЕ ИЗ ИМЕНИ ФАЙЛА
        file_meta = self._get_file_metadata(element)

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
            # 👇 ДОБАВЛЯЕМ НАШИ ПОЛЯ В БАЗОВЫЙ METADATA
            "document_name": file_meta["document_name"],
            "category": file_meta["category"],
            "version": file_meta["version"],
        }

        # Объединяем с существующими метаданными элемента
        meta.update({k: v for k, v in element.metadata.items() if k != "quality"})
        if "quality" in element.metadata:
            meta["quality"] = element.metadata["quality"]

        return meta