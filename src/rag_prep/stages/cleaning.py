"""Очистка текста от шума и boilerplate."""

from __future__ import annotations

import logging
import re
from collections import Counter

from ..config import CleaningConfig
from ..models import ProcessedElement, RawElement

LOGGER = logging.getLogger(__name__)

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


class TextCleaningStage:
    """Удаляет шум, boilerplate и короткие элементы."""

    def __init__(self, config: CleaningConfig):
        self.config = config
        self.drop_patterns = [re.compile(p, re.IGNORECASE) for p in config.drop_patterns]
        self.boilerplate_patterns = [re.compile(p, re.IGNORECASE) for p in config.boilerplate_patterns]

    def run(self, elements: list[RawElement]) -> list[ProcessedElement]:
        cleaned: list[ProcessedElement] = []
        repeat_counts = Counter(self._repeat_key(e.text) for e in elements)

        for element in elements:
            text = self._clean_text(element.text)
            if len(text) < self.config.min_chars:
                continue
            if any(p.search(text) for p in self.drop_patterns):
                continue

            metadata = dict(element.metadata)
            metadata["quality"] = self._quality_signals(text, repeat_counts[self._repeat_key(element.text)])

            cleaned.append(ProcessedElement(
                source_file=element.source_file,
                element_id=element.element_id,
                element_index=element.element_index,
                text=text,
                element_type=element.element_type,
                section=element.section,
                section_path=element.section_path,
                metadata=metadata,
            ))

        LOGGER.info("Cleaned elements: %d -> %d", len(elements), len(cleaned))
        return cleaned

    def _clean_text(self, text: str) -> str:
        cleaned = text.replace("\ufeff", "")
        if self.config.remove_control_chars:
            cleaned = CONTROL_CHARS_RE.sub(" ", cleaned)
        if self.config.normalize_whitespace:
            cleaned = WHITESPACE_RE.sub(" ", cleaned)
        cleaned = BLANK_LINES_RE.sub("\n\n", cleaned)
        return cleaned.strip()

    def _quality_signals(self, text: str, repeated_count: int) -> dict:
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        alpha_chars = sum(1 for c in text if c.isalpha())
        alnum_chars = sum(1 for c in text if c.isalnum())
        printable_chars = sum(1 for c in text if c.isprintable())
        unique_tokens = {t.lower() for t in tokens}

        boilerplate_matches = [p.pattern for p in self.boilerplate_patterns if p.search(text)]
        garbage_score = self._garbage_score(text, alpha_chars, alnum_chars, printable_chars)
        boilerplate_score = min(1.0,
            (0.35 if boilerplate_matches else 0.0) +
            (0.25 if repeated_count > 1 else 0.0) +
            (0.2 if len(tokens) <= 4 else 0.0))
        meaningful_score = max(0.0, min(1.0, 1.0 - max(garbage_score, boilerplate_score)))

        return {
            "meaningful_score": round(meaningful_score, 3),
            "boilerplate_score": round(boilerplate_score, 3),
            "garbage_score": round(garbage_score, 3),
            "is_probable_boilerplate": boilerplate_score >= 0.5,
            "is_probable_garbage": garbage_score >= 0.5,
            "repeated_text_count": repeated_count,
            "unique_token_ratio": round(len(unique_tokens) / max(len(tokens), 1), 3),
            "matched_boilerplate_patterns": boilerplate_matches,
        }

    @staticmethod
    def _repeat_key(text: str) -> str:
        return WHITESPACE_RE.sub(" ", text.strip().lower())

    @staticmethod
    def _garbage_score(text: str, alpha: int, alnum: int, printable: int) -> float:
        length = max(len(text), 1)
        score = 0.0
        if len(text) < 30: score += 0.15
        if alpha / length < 0.35: score += 0.35
        if alnum / length < 0.45: score += 0.25
        score += min(0.25, 1.0 - printable / length)
        return min(1.0, score)