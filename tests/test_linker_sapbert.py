"""Tests for the SapBERT bi-encoder linker.

All dependencies (concept_db, embedding_generator, faiss_index) are mocked
so tests run without model downloads or SNOMED data.
"""

from unittest.mock import MagicMock

import numpy as np

from worker.snomed.data.rf2_parser import SNOMEDConcept
from worker.snomed.linking.sapbert_linker import SEMANTIC_TAG_BOOST, SapBERTLinkerService
from worker.snomed.ner.base import Entity


def _make_entity(text: str = "headache", entity_type: str = "finding") -> Entity:
    return Entity(text=text, start=0, end=len(text), entity_type=entity_type, confidence=0.9)


def _make_concept(
    concept_id: str = "25064002",
    preferred_term: str = "Headache",
    fsn: str = "Headache (finding)",
    semantic_tag: str = "finding",
) -> SNOMEDConcept:
    return SNOMEDConcept(
        concept_id=concept_id,
        preferred_term=preferred_term,
        fsn=fsn,
        synonyms=[],
        semantic_tag=semantic_tag,
        is_active=True,
    )


def _make_mock_deps(
    search_results: list[tuple[str, float]] | None = None,
    concepts: dict[str, SNOMEDConcept] | None = None,
):
    """Create mocked concept_db, embedding_generator, and faiss_index."""
    if search_results is None:
        search_results = [("25064002", 0.95), ("386661006", 0.80)]

    if concepts is None:
        concepts = {
            "25064002": _make_concept("25064002", "Headache", "Headache (finding)", "finding"),
            "386661006": _make_concept("386661006", "Fever", "Fever (finding)", "finding"),
        }

    mock_concept_db = MagicMock()
    mock_concept_db.get_concept.side_effect = lambda cid: concepts.get(cid)

    mock_embedding_gen = MagicMock()
    mock_embedding_gen.embed_single.return_value = np.random.default_rng(42).random(768).astype(np.float32)
    mock_embedding_gen.embed_texts.return_value = np.random.default_rng(42).random((2, 768)).astype(np.float32)

    mock_faiss_index = MagicMock()
    mock_faiss_index.search_single.return_value = search_results
    # For batch search: return arrays matching the batch shape
    n_queries = 2
    distances = np.array([[s for _, s in search_results]] * n_queries, dtype=np.float32)
    concept_ids = [[cid for cid, _ in search_results]] * n_queries
    mock_faiss_index.search.return_value = (distances, concept_ids)

    return mock_concept_db, mock_embedding_gen, mock_faiss_index


class TestSapBERTLinkerService:
    def test_link_entity_basic(self):
        """Basic entity linking returns top concept and alternatives."""
        concept_db, embedding_gen, faiss_index = _make_mock_deps()
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
            confidence_threshold=0.5,
        )

        result = linker.link_entity(_make_entity())

        assert result.top_concept is not None
        assert result.top_concept.concept_id == "25064002"
        assert result.top_concept.preferred_term == "Headache"
        assert len(result.alternatives) == 2
        embedding_gen.embed_single.assert_called_once_with("headache")
        faiss_index.search_single.assert_called_once()

    def test_link_entity_confidence_filtering(self):
        """Candidates below confidence threshold are filtered out."""
        concept_db, embedding_gen, faiss_index = _make_mock_deps(
            search_results=[("25064002", 0.95), ("386661006", 0.30)]
        )
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
            confidence_threshold=0.7,
        )

        result = linker.link_entity(_make_entity())

        # Only headache (0.95 + boost) should pass; fever (0.30 + boost) should not
        assert result.top_concept is not None
        assert result.top_concept.concept_id == "25064002"
        assert all(c.score >= 0.7 for c in result.alternatives)

    def test_link_entity_no_results(self):
        """When no candidates pass threshold, top_concept is None."""
        concept_db, embedding_gen, faiss_index = _make_mock_deps(search_results=[("25064002", 0.3)])
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
            confidence_threshold=0.9,
        )

        result = linker.link_entity(_make_entity())

        assert result.top_concept is None
        assert result.alternatives == []

    def test_semantic_tag_boost(self):
        """Matching semantic tags get a score boost."""
        concepts = {
            "25064002": _make_concept("25064002", "Headache", "Headache (finding)", "finding"),
            "80891009": _make_concept("80891009", "Heart", "Heart structure (body structure)", "body_structure"),
        }
        concept_db, embedding_gen, faiss_index = _make_mock_deps(
            search_results=[("80891009", 0.90), ("25064002", 0.88)],
            concepts=concepts,
        )
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
            confidence_threshold=0.5,
            filter_by_semantic_tag=True,
        )

        entity = _make_entity("headache", entity_type="finding")
        result = linker.link_entity(entity)

        # Headache (finding) should be boosted above Heart (body_structure)
        assert result.top_concept is not None
        assert result.top_concept.concept_id == "25064002"
        # Heart gets penalised, Headache gets boosted
        headache = next(c for c in result.alternatives if c.concept_id == "25064002")
        heart = next(c for c in result.alternatives if c.concept_id == "80891009")
        assert headache.score == 0.88 + SEMANTIC_TAG_BOOST
        assert heart.score == 0.90 - SEMANTIC_TAG_BOOST

    def test_semantic_tag_filter_disabled(self):
        """When filter_by_semantic_tag=False, no score adjustment is made."""
        concepts = {
            "25064002": _make_concept("25064002", "Headache", "Headache (finding)", "finding"),
            "80891009": _make_concept("80891009", "Heart", "Heart structure (body structure)", "body_structure"),
        }
        concept_db, embedding_gen, faiss_index = _make_mock_deps(
            search_results=[("80891009", 0.90), ("25064002", 0.88)],
            concepts=concepts,
        )
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
            confidence_threshold=0.5,
            filter_by_semantic_tag=False,
        )

        result = linker.link_entity(_make_entity("headache", entity_type="finding"))

        # Without filter, Heart (0.90) stays on top
        assert result.top_concept is not None
        assert result.top_concept.concept_id == "80891009"
        assert result.top_concept.score == 0.90

    def test_link_entity_missing_concept(self):
        """Concepts not found in the database are skipped."""
        concepts = {"25064002": _make_concept("25064002", "Headache", "Headache (finding)", "finding")}
        concept_db, embedding_gen, faiss_index = _make_mock_deps(
            search_results=[("MISSING_ID", 0.95), ("25064002", 0.88)],
            concepts=concepts,
        )
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
            confidence_threshold=0.5,
        )

        result = linker.link_entity(_make_entity())

        # MISSING_ID skipped, only Headache returned
        assert len(result.alternatives) == 1
        assert result.top_concept.concept_id == "25064002"

    def test_link_entities_batch(self):
        """Batch linking uses batched embedding and search."""
        concept_db, embedding_gen, faiss_index = _make_mock_deps()
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
            confidence_threshold=0.5,
        )

        entities = [_make_entity("headache"), _make_entity("fever")]
        results = linker.link_entities(entities)

        assert len(results) == 2
        embedding_gen.embed_texts.assert_called_once_with(["headache", "fever"])
        faiss_index.search.assert_called_once()
        # embed_single should NOT be called (batch path used)
        embedding_gen.embed_single.assert_not_called()

    def test_link_entities_empty(self):
        """Empty input returns empty output."""
        concept_db, embedding_gen, faiss_index = _make_mock_deps()
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
        )

        results = linker.link_entities([])
        assert results == []

    def test_link_entity_preserves_entity(self):
        """The original entity is preserved in the LinkResult."""
        concept_db, embedding_gen, faiss_index = _make_mock_deps()
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
            confidence_threshold=0.5,
        )

        entity = _make_entity("headache")
        result = linker.link_entity(entity)

        assert result.entity is entity
        assert result.entity.text == "headache"

    def test_custom_top_k(self):
        """Custom top_k is passed to FAISS search."""
        concept_db, embedding_gen, faiss_index = _make_mock_deps()
        linker = SapBERTLinkerService(
            concept_db=concept_db,
            embedding_generator=embedding_gen,
            faiss_index=faiss_index,
            confidence_threshold=0.5,
        )

        linker.link_entity(_make_entity(), top_k=10)
        faiss_index.search_single.assert_called_once()
        _, kwargs = faiss_index.search_single.call_args
        assert kwargs.get("top_k") == 10

    def test_uses_settings_defaults(self):
        """Service uses settings values when not overridden."""
        from unittest.mock import patch

        concept_db, embedding_gen, faiss_index = _make_mock_deps()
        with patch("worker.snomed.linking.sapbert_linker.get_settings") as mock_get_settings:
            mock_get_settings.return_value.SNOMED_CONFIDENCE_THRESHOLD = 0.7
            mock_get_settings.return_value.SNOMED_TOP_K_CANDIDATES = 5
            linker = SapBERTLinkerService(
                concept_db=concept_db,
                embedding_generator=embedding_gen,
                faiss_index=faiss_index,
            )
        assert linker.confidence_threshold == 0.7
        assert linker.default_top_k == 5
        assert linker.filter_by_semantic_tag is True
