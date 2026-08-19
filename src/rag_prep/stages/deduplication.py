"""Дедупликация через MinHash LSH."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from datasketch import MinHash, MinHashLSH

from ..config import DeduplicationConfig
from ..models import ProcessedElement
from ..utils import text_sha256

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class DeduplicationResult:
    elements: list[ProcessedElement]
    duplicates_removed: int
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0


class DeduplicationStage:
    """Удаляет exact и near-дубликаты."""

    def __init__(self, config: DeduplicationConfig):
        self.config = config

    def run(self, elements: list[ProcessedElement]) -> DeduplicationResult:
        if not self.config.enabled:
            return DeduplicationResult(elements=elements, duplicates_removed=0)

        seen_hashes: set[str] = set()
        lsh = MinHashLSH(threshold=self.config.threshold, num_perm=self.config.num_perm)
        kept: list[ProcessedElement] = []
        inserted_keys: set[str] = set()
        kept_short: list[str] = []
        exact_dup = 0
        near_dup = 0

        for pos, element in enumerate(elements):
            digest = text_sha256(element.text)
            if digest in seen_hashes:
                exact_dup += 1
                continue

            tokens = self._tokens(element.text)
            if len(tokens) < self.config.min_tokens:
                norm_short = " ".join(tokens)
                if self._is_near_short_duplicate(norm_short, kept_short):
                    seen_hashes.add(digest)
                    near_dup += 1
                    continue
                kept_short.append(norm_short)
            else:
                minhash = self._minhash(tokens)
                if lsh.query(minhash):
                    seen_hashes.add(digest)
                    near_dup += 1
                    continue
                key = self._lsh_key(pos, digest, inserted_keys)
                lsh.insert(key, minhash)
                inserted_keys.add(key)

            seen_hashes.add(digest)
            kept.append(element)

        total_dup = exact_dup + near_dup
        LOGGER.info("Deduplication: %d -> %d; exact=%d near=%d",
                    len(elements), len(kept), exact_dup, near_dup)

        return DeduplicationResult(
            elements=kept,
            duplicates_removed=total_dup,
            exact_duplicates_removed=exact_dup,
            near_duplicates_removed=near_dup,
        )

    @staticmethod
    def _lsh_key(position: int, digest: str, inserted: set[str]) -> str:
        key = f"{position}:{digest}"
        if key not in inserted:
            return key
        suffix = 1
        while f"{key}:{suffix}" in inserted:
            suffix += 1
        return f"{key}:{suffix}"

    def _tokens(self, text: str) -> list[str]:
        return [t.lower() for t in TOKEN_RE.findall(text)]

    def _is_near_short_duplicate(self, text: str, candidates: list[str]) -> bool:
        if not text:
            return False
        return any(SequenceMatcher(None, text, c).ratio() >= self.config.threshold for c in candidates)

    def _minhash(self, tokens: list[str]) -> MinHash:
        m = MinHash(num_perm=self.config.num_perm)
        for shingle in self._shingles(tokens):
            m.update(" ".join(shingle).encode("utf-8"))
        return m

    def _shingles(self, tokens: list[str]) -> list[tuple[str, ...]]:
        size = self.config.shingle_size
        if len(tokens) < size:
            return [tuple(tokens)]
        return [tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]