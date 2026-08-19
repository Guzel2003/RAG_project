"""Нормализация текста без spaCy (лёгкая версия)."""

from __future__ import annotations

import logging
import re
import unicodedata

from ..config import NormalizationConfig
from ..models import ProcessedElement

LOGGER = logging.getLogger(__name__)

SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')


class TextNormalizationStage:
    """Нормализует Unicode и считает предложения через regex."""

    def __init__(self, config: NormalizationConfig):
        self.config = config

    def run(self, elements: list[ProcessedElement]) -> list[ProcessedElement]:
        normalized: list[ProcessedElement] = []

        for element in elements:
            text = self._normalize_text(element.text)
            metadata = dict(element.metadata)

            if self.config.collect_sentence_stats:
                sentences = [s for s in SENTENCE_ENDINGS.split(text) if s.strip()]
                tokens = re.findall(r'\w+', text, flags=re.UNICODE)
                metadata["sentence_count"] = len(sentences)
                metadata["token_count"] = len(tokens)

            normalized.append(element.model_copy(update={"text": text, "metadata": metadata}))

        LOGGER.info("Normalized %d elements", len(normalized))
        return normalized

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize(self.config.unicode_form, text)
        if self.config.lowercase:
            normalized = normalized.lower()
        return normalized