"""Tests for SnomedConceptSearchService."""

import json
from pathlib import Path

import pytest

from common.services.snomed_search_service import SnomedConceptSearchService
from common.snomed_types import SNOMEDConcept
from tests.fixtures.snomed_mock import MOCK_CONCEPTS


@pytest.fixture()
def concept_db_path(tmp_path: Path) -> str:
    """Create a temporary concept_db.json from mock data."""
    concepts = {}
    for data in MOCK_CONCEPTS:
        concept = SNOMEDConcept(
            concept_id=data["concept_id"],
            preferred_term=data["preferred_term"],
            fsn=data["fsn"],
            synonyms=data["synonyms"],
            semantic_tag=data["semantic_tag"],
            is_active=True,
            parent_ids=[],
        )
        concepts[concept.concept_id] = concept.to_dict()

    db_file = tmp_path / "concept_db.json"
    db_file.write_text(json.dumps({"concepts": concepts}), encoding="utf-8")
    return str(tmp_path)


@pytest.fixture()
def search_service(concept_db_path: str) -> SnomedConceptSearchService:
    """Create a fresh search service instance loaded with mock data."""
    # Don't use singleton to avoid test interference
    service = SnomedConceptSearchService()
    service.load(concept_db_path)
    return service


class TestSnomedConceptSearchService:
    def test_load_concepts(self, search_service: SnomedConceptSearchService) -> None:
        assert len(search_service.concepts) == len(MOCK_CONCEPTS)
        assert search_service._loaded is True  # noqa: SLF001

    def test_load_missing_file(self, tmp_path: Path) -> None:
        service = SnomedConceptSearchService()
        service.load(str(tmp_path / "nonexistent"))
        assert len(service.concepts) == 0
        assert service._loaded is False  # noqa: SLF001

    def test_load_is_idempotent(self, concept_db_path: str) -> None:
        service = SnomedConceptSearchService()
        service.load(concept_db_path)
        count = len(service.concepts)
        service.load(concept_db_path)
        assert len(service.concepts) == count

    def test_exact_search(self, search_service: SnomedConceptSearchService) -> None:
        results = search_service.search("Headache")
        assert len(results) >= 1
        concept_ids = [r.concept_id for r in results]
        assert "25064002" in concept_ids

    def test_prefix_search(self, search_service: SnomedConceptSearchService) -> None:
        results = search_service.search("head", limit=10)
        assert len(results) >= 1
        # Should find "Headache" via prefix match on "headache", "head pain"
        concept_ids = [r.concept_id for r in results]
        assert "25064002" in concept_ids

    def test_synonym_search(self, search_service: SnomedConceptSearchService) -> None:
        results = search_service.search("cephalalgia")
        assert len(results) >= 1
        concept_ids = [r.concept_id for r in results]
        assert "25064002" in concept_ids  # Headache has synonym "Cephalalgia"

    def test_case_insensitive(self, search_service: SnomedConceptSearchService) -> None:
        results_lower = search_service.search("headache")
        results_upper = search_service.search("HEADACHE")
        assert len(results_lower) == len(results_upper)

    def test_empty_query(self, search_service: SnomedConceptSearchService) -> None:
        results = search_service.search("")
        assert results == []

    def test_whitespace_query(self, search_service: SnomedConceptSearchService) -> None:
        results = search_service.search("   ")
        assert results == []

    def test_no_results(self, search_service: SnomedConceptSearchService) -> None:
        results = search_service.search("xyznonexistent")
        assert results == []

    def test_limit(self, search_service: SnomedConceptSearchService) -> None:
        results = search_service.search("a", limit=3)
        assert len(results) <= 3

    def test_get_concept(self, search_service: SnomedConceptSearchService) -> None:
        concept = search_service.get_concept("25064002")
        assert concept is not None
        assert concept.preferred_term == "Headache"

    def test_get_concept_not_found(self, search_service: SnomedConceptSearchService) -> None:
        concept = search_service.get_concept("999999999")
        assert concept is None

    def test_exact_match_sorted_first(self, search_service: SnomedConceptSearchService) -> None:
        results = search_service.search("headache", limit=10)
        assert len(results) >= 1
        # Exact preferred_term match should be first
        assert results[0].preferred_term == "Headache"

    def test_search_by_abbreviation(self, search_service: SnomedConceptSearchService) -> None:
        results = search_service.search("MRI")
        assert len(results) >= 1
        concept_ids = [r.concept_id for r in results]
        assert "241615005" in concept_ids  # Magnetic resonance imaging

    def test_singleton_pattern(self) -> None:
        # Reset singleton
        SnomedConceptSearchService._instance = None  # noqa: SLF001
        instance1 = SnomedConceptSearchService.get_instance()
        instance2 = SnomedConceptSearchService.get_instance()
        assert instance1 is instance2
        # Clean up
        SnomedConceptSearchService._instance = None  # noqa: SLF001
