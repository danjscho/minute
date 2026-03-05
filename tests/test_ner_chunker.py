"""Tests for the NER text chunker."""

import pytest

from worker.snomed.ner.base import Entity
from worker.snomed.ner.chunker import TextChunk, TextChunker


class TestTextChunker:
    def test_short_text_single_chunk(self):
        """Text within token limit should produce a single chunk."""
        chunker = TextChunker(max_tokens=512)
        text = "Patient has headache and fever."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].offset == 0

    def test_empty_text(self):
        chunker = TextChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_long_text_multiple_chunks(self):
        """Text exceeding token limit should be split into multiple chunks."""
        chunker = TextChunker(max_tokens=20, overlap_tokens=5)
        # Create text with multiple sentences that exceeds 20 tokens (~100 chars)
        sentences = [
            "Patient presents with severe headache.",
            "Blood pressure is elevated.",
            "Temperature is normal.",
            "Heart rate is regular.",
            "Respiratory rate within normal limits.",
        ]
        text = " ".join(sentences)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunks_cover_full_text(self):
        """All text should be covered by at least one chunk."""
        chunker = TextChunker(max_tokens=15, overlap_tokens=3)
        sentences = [
            "Headache reported.",
            "Fever present.",
            "Cough noted.",
            "MRI scheduled.",
        ]
        text = " ".join(sentences)
        chunks = chunker.chunk_text(text)

        # Every character in the original text should appear in at least one chunk
        covered = set()
        for chunk in chunks:
            for i in range(len(chunk.text)):
                covered.add(chunk.offset + i)

        for i in range(len(text)):
            if text[i].strip():  # Skip whitespace that may be at boundaries
                assert i in covered or text[i] == " ", f"Character at position {i} not covered: '{text[i]}'"

    def test_chunk_offsets_correct(self):
        """Chunk offsets should point to correct positions in original text."""
        chunker = TextChunker(max_tokens=512)
        text = "This is a simple test."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].offset == 0
        assert text[chunks[0].offset : chunks[0].offset + len(chunks[0].text)] == text

    def test_sentence_boundary_splitting(self):
        """Chunks should split at sentence boundaries when possible."""
        chunker = TextChunker(max_tokens=10, overlap_tokens=2)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_text(text)
        # Each chunk should end at or near a sentence boundary
        for chunk in chunks:
            # Chunks should contain complete sentences (not split mid-word)
            assert not chunk.text.startswith(" "), f"Chunk starts with space: '{chunk.text}'"


class TestTextChunkDataclass:
    def test_text_chunk_creation(self):
        chunk = TextChunk(text="hello world", offset=10, approx_tokens=2)
        assert chunk.text == "hello world"
        assert chunk.offset == 10
        assert chunk.approx_tokens == 2

    def test_text_chunk_default_tokens(self):
        chunk = TextChunk(text="test", offset=0)
        assert chunk.approx_tokens == 0


class TestEntityMerging:
    def test_no_overlap_preserves_all(self):
        """Entities from non-overlapping regions should all be preserved."""
        chunker = TextChunker()
        chunks = [
            TextChunk(text="Headache noted.", offset=0),
            TextChunk(text="Fever present.", offset=50),
        ]
        chunk_entities = [
            [Entity(text="Headache", start=0, end=8, entity_type="finding", confidence=0.9)],
            [Entity(text="Fever", start=0, end=5, entity_type="finding", confidence=0.85)],
        ]
        merged = chunker.merge_entities(chunk_entities, chunks)
        assert len(merged) == 2
        # Offsets should be adjusted
        assert merged[0].start == 0  # 0 + chunk offset 0
        assert merged[1].start == 50  # 0 + chunk offset 50

    def test_overlapping_keeps_highest_confidence(self):
        """When entities overlap, the highest confidence one wins."""
        chunker = TextChunker()
        chunks = [
            TextChunk(text="headache and fever", offset=10),
            TextChunk(text="headache and fever", offset=10),  # Same overlap region
        ]
        chunk_entities = [
            [Entity(text="headache", start=0, end=8, entity_type="finding", confidence=0.7)],
            [Entity(text="headache", start=0, end=8, entity_type="finding", confidence=0.9)],
        ]
        merged = chunker.merge_entities(chunk_entities, chunks)
        assert len(merged) == 1
        assert merged[0].confidence == 0.9

    def test_partial_overlap_dedup(self):
        """Partially overlapping entities should be deduplicated."""
        chunker = TextChunker()
        chunks = [
            TextChunk(text="severe headache", offset=0),
            TextChunk(text="headache and nausea", offset=7),
        ]
        chunk_entities = [
            [Entity(text="severe headache", start=0, end=15, entity_type="finding", confidence=0.9)],
            [Entity(text="headache", start=0, end=8, entity_type="finding", confidence=0.8)],
        ]
        merged = chunker.merge_entities(chunk_entities, chunks)
        # Should keep only the higher-confidence one since they overlap
        assert len(merged) == 1
        assert merged[0].confidence == 0.9

    def test_empty_entities(self):
        chunker = TextChunker()
        chunks = [TextChunk(text="No findings.", offset=0)]
        chunk_entities: list[list[Entity]] = [[]]
        merged = chunker.merge_entities(chunk_entities, chunks)
        assert merged == []

    def test_mismatched_lengths_raises(self):
        chunker = TextChunker()
        with pytest.raises(ValueError, match="same length"):
            chunker.merge_entities([[]], [TextChunk(text="a", offset=0), TextChunk(text="b", offset=5)])

    def test_merged_sorted_by_start(self):
        """Merged entities should be sorted by start position."""
        chunker = TextChunker()
        chunks = [
            TextChunk(text="fever and cough", offset=50),
            TextChunk(text="headache noted", offset=0),
        ]
        chunk_entities = [
            [Entity(text="fever", start=0, end=5, entity_type="finding", confidence=0.9)],
            [Entity(text="headache", start=0, end=8, entity_type="finding", confidence=0.85)],
        ]
        merged = chunker.merge_entities(chunk_entities, chunks)
        assert len(merged) == 2
        assert merged[0].start < merged[1].start
