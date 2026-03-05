"""
Text chunking for NER on long documents.

Splits text into overlapping chunks at sentence boundaries so that
the NER model's token limit (typically 512) is respected. Entities
from overlapping regions are merged by keeping the highest-confidence
prediction for each span.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worker.snomed.ner.base import Entity

logger = logging.getLogger(__name__)

# Sentence-ending punctuation followed by whitespace
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Rough chars-per-token estimate for English clinical text
_CHARS_PER_TOKEN_ESTIMATE = 5


@dataclass
class TextChunk:
    """A chunk of text with its offset in the original document.

    Attributes:
        text: The chunk text.
        offset: Character offset of this chunk's start in the original text.
        approx_tokens: Approximate token count (character-based estimate).
    """

    text: str
    offset: int
    approx_tokens: int = 0


@dataclass
class TextChunker:
    """Split long text into overlapping chunks at sentence boundaries.

    Attributes:
        max_tokens: Maximum approximate tokens per chunk.
        overlap_tokens: Number of tokens of overlap between consecutive chunks.
    """

    max_tokens: int = 512
    overlap_tokens: int = 50

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character length."""
        return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = _SENTENCE_SPLIT_RE.split(text)
        return [s for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> list[TextChunk]:
        """Split text into overlapping chunks at sentence boundaries.

        If the text fits within max_tokens, returns a single chunk.
        Otherwise, groups sentences into chunks with overlap.

        Args:
            text: Full input text.

        Returns:
            List of TextChunk objects with correct offsets.
        """
        if not text.strip():
            return []

        total_tokens = self._estimate_tokens(text)
        if total_tokens <= self.max_tokens:
            return [TextChunk(text=text, offset=0, approx_tokens=total_tokens)]

        sentences = self._split_sentences(text)
        if not sentences:
            return [TextChunk(text=text, offset=0, approx_tokens=total_tokens)]

        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        current_offset = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            if current_tokens + sent_tokens > self.max_tokens and current_sentences:
                # Emit current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        offset=current_offset,
                        approx_tokens=current_tokens,
                    )
                )

                # Calculate overlap: walk backwards through sentences
                overlap_sentences, overlap_token_count = self._compute_overlap(current_sentences)

                # Start new chunk with overlap sentences
                new_offset = current_offset + len(chunk_text) - len(" ".join(overlap_sentences))
                current_sentences = list(overlap_sentences)
                current_tokens = overlap_token_count
                current_offset = new_offset

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Emit final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    offset=current_offset,
                    approx_tokens=current_tokens,
                )
            )

        logger.info("Split text (%d chars) into %d chunks", len(text), len(chunks))
        return chunks

    def _compute_overlap(self, sentences: list[str]) -> tuple[list[str], int]:
        """Compute overlap sentences from the end of a chunk.

        Returns:
            Tuple of (overlap_sentences, overlap_token_count).
        """
        overlap_sentences: list[str] = []
        overlap_tokens = 0

        for sentence in reversed(sentences):
            sent_tokens = self._estimate_tokens(sentence)
            if overlap_tokens + sent_tokens > self.overlap_tokens and overlap_sentences:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sent_tokens

        return overlap_sentences, overlap_tokens

    def merge_entities(
        self,
        chunk_entities: list[list[Entity]],
        chunks: list[TextChunk],
    ) -> list[Entity]:
        """Merge entities from overlapping chunks.

        Adjusts character offsets to the original text coordinates and
        deduplicates entities from overlap regions by keeping the
        highest-confidence prediction for overlapping spans.

        Args:
            chunk_entities: List of entity lists, one per chunk.
            chunks: Corresponding TextChunk objects with offset info.

        Returns:
            Deduplicated list of entities with original-text offsets, sorted by start position.
        """
        from worker.snomed.ner.base import Entity

        if len(chunk_entities) != len(chunks):
            msg = "chunk_entities and chunks must have the same length"
            raise ValueError(msg)

        # Adjust offsets to original text coordinates
        all_entities: list[Entity] = []
        for entities, chunk in zip(chunk_entities, chunks, strict=False):
            for entity in entities:
                adjusted = Entity(
                    text=entity.text,
                    start=entity.start + chunk.offset,
                    end=entity.end + chunk.offset,
                    entity_type=entity.entity_type,
                    confidence=entity.confidence,
                    raw_label=entity.raw_label,
                )
                all_entities.append(adjusted)

        if not all_entities:
            return []

        # Sort by start position, then by confidence descending
        all_entities.sort(key=lambda e: (e.start, -e.confidence))

        # Deduplicate overlapping spans: keep highest confidence
        merged: list[Entity] = []
        for entity in all_entities:
            if self._overlaps_existing(entity, merged):
                continue
            merged.append(entity)

        merged.sort(key=lambda e: e.start)
        return merged

    def _overlaps_existing(self, entity: Entity, existing: list[Entity]) -> bool:
        """Check if entity overlaps with any existing entity.

        Two spans overlap if they share any characters. When overlap is found,
        the existing entity (already in the list) is kept if it has higher
        confidence; otherwise it is replaced.
        """
        for i, existing_entity in enumerate(existing):
            if entity.start < existing_entity.end and entity.end > existing_entity.start:
                # Overlap detected — keep higher confidence
                if entity.confidence > existing_entity.confidence:
                    existing[i] = entity
                return True
        return False
