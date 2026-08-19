"""Semantic-aware chunking."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import tiktoken
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

from ..config_chunking import ChunkingConfig
from ..models_chunking import ChunkMetadata, PreparedChunk
from ..models import PreparedDocument
from ..utils import stable_id, text_sha256

LOGGER = logging.getLogger(__name__)

_PARAGRAPH_SEPARATOR_RE = re.compile(r"(?:\r?\n[ \t]*){2,}")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?]+")


@dataclass(frozen=True)
class SemanticBlock:
    id: str
    text: str
    position: int
    start_char: int
    end_char: int
    token_count: int


class ChunkSplittingStage:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.encoding = self._init_tokenizer()
        self.splitter = self._build_splitter()
        self._run_id = "standalone"

    def _init_tokenizer(self):
        try: return tiktoken.encoding_for_model(self.config.tokenizer_model)
        except KeyError:
            try: return tiktoken.get_encoding(self.config.tokenizer_model)
            except ValueError:
                LOGGER.warning("Неизвестная модель %s; используем cl100k_base", self.config.tokenizer_model)
                return tiktoken.get_encoding("cl100k_base")

    def _build_splitter(self):
        kwargs = {"chunk_size": self.config.chunk_size, "chunk_overlap": self.config.chunk_overlap, "tokenizer": self._tokenize}
        return TokenTextSplitter(**kwargs) if self.config.strategy == "token" else SentenceSplitter(**kwargs)

    def _tokenize(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def run(self, documents: list[PreparedDocument], run_id: str = "standalone") -> list[PreparedChunk]:
        self._run_id = run_id
        chunks: list[PreparedChunk] = []
        for doc in documents: chunks.extend(self._split_document(doc))
        LOGGER.info("Разделено документов: %d; получено чанков: %d", len(documents), len(chunks))
        return chunks

    def _split_document(self, document: PreparedDocument) -> list[PreparedChunk]:
        if not document.text.strip(): return []
        blocks = self._semantic_blocks(document)
        if self.config.preserve_section_boundaries and self.config.preserve_block_boundaries:
            return self._split_semantic_blocks(document, blocks)
        return self._split_whole_document(document, blocks)

    def _semantic_blocks(self, document: PreparedDocument) -> list[SemanticBlock]:
        spans = self._paragraph_spans(document.text)
        blocks: list[SemanticBlock] = []
        for pos, (text, start, end) in enumerate(spans):
            block_id = stable_id(document.metadata["document_id"], "block", pos, start, end, text_sha256(text))
            blocks.append(SemanticBlock(id=block_id, text=text, position=pos, start_char=start, end_char=end, token_count=len(self._tokenize(text))))
        return blocks

    @staticmethod
    def _paragraph_spans(text: str) -> list[tuple[str, int, int]]:
        blocks: list[tuple[str, int, int]] = []
        cursor = 0
        for sep in _PARAGRAPH_SEPARATOR_RE.finditer(text):
            seg = text[cursor:sep.start()]
            if seg.strip():
                lead = len(seg) - len(seg.lstrip()); trail = len(seg.rstrip())
                blocks.append((seg.strip(), cursor + lead, cursor + trail))
            cursor = sep.end()
        seg = text[cursor:]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip()); trail = len(seg.rstrip())
            blocks.append((seg.strip(), cursor + lead, cursor + trail))
        return blocks

    def _split_semantic_blocks(self, document: PreparedDocument, blocks: list[SemanticBlock]) -> list[PreparedChunk]:
        chunks: list[PreparedChunk] = []
        current: list[SemanticBlock] = []
        for block in blocks:
            if block.token_count > self.config.chunk_size:
                if current: chunks.append(self._chunk_from_blocks(document, current, len(chunks)))
                current = []
                chunks.extend(self._split_oversized_block(document, block, len(chunks)))
                continue
            candidate = [*current, block]
            if current and self._joined_token_count(candidate) > self.config.chunk_size:
                chunks.append(self._chunk_from_blocks(document, current, len(chunks)))
                overlap = self._overlap_blocks(current, next_block=block)
                current = [*overlap, block]
            else: current = candidate
        if current: chunks.append(self._chunk_from_blocks(document, current, len(chunks)))
        return chunks

    def _split_whole_document(self, document: PreparedDocument, blocks: list[SemanticBlock]) -> list[PreparedChunk]:
        split_texts = self.splitter.split_text(document.text)
        chunks: list[PreparedChunk] = []
        cursor = 0
        for chunk_text in split_texts:
            norm = chunk_text.strip()
            if not norm: continue
            start = document.text.find(norm, max(0, cursor - self.config.chunk_overlap * 4))
            if start == -1: start = cursor
            end = start + len(norm); cursor = end
            chunks.append(self._chunk_from_text(document=document, chunk_text=norm, position=len(chunks), start_char=start, end_char=end, blocks=self._blocks_for_span(blocks, start, end), offset_strategy="exact" if start != cursor - len(norm) else "estimated"))
        return chunks

    def _split_oversized_block(self, document: PreparedDocument, block: SemanticBlock, start_pos: int) -> list[PreparedChunk]:
        split_texts = self.splitter.split_text(block.text)
        chunks: list[PreparedChunk] = []
        cursor = 0
        for chunk_text in split_texts:
            norm = chunk_text.strip()
            if not norm: continue
            start = block.text.find(norm, cursor)
            if start == -1: start = cursor
            end = start + len(norm); cursor = end
            chunks.append(self._chunk_from_text(document=document, chunk_text=norm, position=start_pos + len(chunks), start_char=block.start_char + start, end_char=block.start_char + end, blocks=[block], offset_strategy="semantic_block_span"))
        return chunks

    def _chunk_from_blocks(self, document: PreparedDocument, blocks: list[SemanticBlock], position: int) -> PreparedChunk:
        return self._chunk_from_text(document=document, chunk_text="\n\n".join(b.text for b in blocks).strip(), position=position, start_char=blocks[0].start_char, end_char=blocks[-1].end_char, blocks=blocks, offset_strategy="semantic_block_span")

    def _chunk_from_text(self, *, document: PreparedDocument, chunk_text: str, position: int, start_char: int, end_char: int, blocks: list[SemanticBlock], offset_strategy: str) -> PreparedChunk:
        doc_meta = document.metadata
        token_count = len(self._tokenize(chunk_text))
        text_hash = text_sha256(chunk_text)
        chunk_id = stable_id(doc_meta["document_id"], position, start_char, end_char, text_hash)
        lineage = {"document_id": doc_meta["document_id"], "chunk_id": chunk_id, "chunk_position": position, "pipeline_stage": "chunk", "chunking_run_id": self._run_id}
        quality = self._quality(chunk_text, token_count, blocks)
        metadata = ChunkMetadata(
            id=chunk_id, document_id=doc_meta["document_id"], source=doc_meta.get("source", ""), section=doc_meta.get("section", "full_document"),
            position=position, chunk_start_char=start_char, chunk_end_char=end_char, chunk_token_count=token_count, chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap, chunking_strategy=self.config.strategy, tokenizer_model=self.config.tokenizer_model,
            embedding_model=self.config.embedding_model, chunking_run_id=self._run_id, semantic_block_ids=[b.id for b in blocks],
            semantic_block_start=min(b.position for b in blocks) if blocks else None, semantic_block_end=max(b.position for b in blocks) if blocks else None,
            offset_strategy=offset_strategy, parent_ids=[doc_meta["document_id"]], origin_element_ids=[], lineage=lineage,
            hierarchy={"chunk_position": position, "semantic_block_count": len(blocks)}, source_hash=doc_meta.get("source_hash", ""),
            document_text_hash=doc_meta.get("source_sha256", ""), text_hash=text_hash, file_name=doc_meta.get("file_name", ""), file_type=doc_meta.get("file_type", ""), quality=quality,
        )
        return PreparedChunk(text=chunk_text, metadata=metadata)

    def _overlap_blocks(self, blocks: list[SemanticBlock], next_block: SemanticBlock) -> list[SemanticBlock]:
        if self.config.chunk_overlap <= 0: return []
        overlap: list[SemanticBlock] = []
        for block in reversed(blocks):
            candidate = [block, *overlap]
            if self._joined_token_count(candidate) <= self.config.chunk_overlap: overlap = candidate
            else: break
        return overlap if overlap and self._joined_token_count([*overlap, next_block]) <= self.config.chunk_size else []

    def _joined_token_count(self, blocks: list[SemanticBlock]) -> int:
        return len(self._tokenize("\n\n".join(b.text for b in blocks)))

    @staticmethod
    def _blocks_for_span(blocks: list[SemanticBlock], start: int, end: int) -> list[SemanticBlock]:
        return [b for b in blocks if b.start_char < end and b.end_char > start]

    def _quality(self, text: str, token_count: int, blocks: list[SemanticBlock]) -> dict:
        words = [w.lower() for w in _WORD_RE.findall(text)]
        char_count = len(text)
        alpha = sum(1 for c in text if c.isalpha())
        known = sum(1 for c in text if self._is_supported_letter(c))
        noisy = sum(1 for c in text if self._is_noise_char(c))
        sentences = len(_SENTENCE_END_RE.findall(text))
        unique_ratio = len(set(words)) / len(words) if words else 0.0
        lang_conf = known / alpha if alpha else 0.0
        ocr_noise = min(1.0, noisy / char_count) if char_count else 0.0
        length_score = min(1.0, token_count / self.config.min_chunk_tokens)
        sent_score = 1.0 if sentences > 0 or len(blocks) > 1 else 0.5
        block_score = min(1.0, len(blocks) / 2) if blocks else 0.0
        structure = max(0.0, min(1.0, 0.35 * length_score + 0.25 * sent_score + 0.20 * block_score + 0.20 * unique_ratio - 0.35 * ocr_noise))
        return {"token_density": round(token_count / char_count, 4) if char_count else 0.0, "language_confidence": round(lang_conf, 3), "ocr_noise_score": round(ocr_noise, 3), "structure_score": round(structure, 3), "unique_token_ratio": round(unique_ratio, 3), "semantic_block_count": len(blocks), "sentence_count": sentences, "char_count": char_count, "word_count": len(words), "is_low_quality_chunk": structure < self.config.min_quality_score}

    @staticmethod
    def _is_supported_letter(char: str) -> bool:
        lower = char.lower()
        return ("а" <= lower <= "я") or lower == "ё" or ("a" <= lower <= "z")

    @staticmethod
    def _is_noise_char(char: str) -> bool:
        if char == "\ufffd" or (ord(char) < 32 and char not in "\n\r\t"): return True
        common = set(".,;:!?()[]{}<>«»\"'`%-–—/\\№+=_*&@#$|")
        return not char.isalnum() and not char.isspace() and char not in common