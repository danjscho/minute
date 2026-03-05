"""Tests for ConceptDatabase."""

from common.snomed_types import SNOMEDConcept
from worker.snomed.data.concept_db import ConceptDatabase


def _make_concepts() -> dict[str, SNOMEDConcept]:
    return {
        "25064002": SNOMEDConcept(
            concept_id="25064002",
            preferred_term="Headache",
            fsn="Headache (finding)",
            synonyms=["Cephalalgia", "Head pain"],
            semantic_tag="finding",
        ),
        "241615005": SNOMEDConcept(
            concept_id="241615005",
            preferred_term="Magnetic resonance imaging",
            fsn="Magnetic resonance imaging (procedure)",
            synonyms=["MRI", "MRI scan"],
            semantic_tag="procedure",
        ),
        "80891009": SNOMEDConcept(
            concept_id="80891009",
            preferred_term="Heart",
            fsn="Heart structure (body structure)",
            synonyms=["Cardiac structure"],
            semantic_tag="body_structure",
        ),
    }


class TestConceptDatabase:
    def test_init_empty(self) -> None:
        db = ConceptDatabase()
        assert len(db.concepts) == 0

    def test_init_with_concepts(self) -> None:
        db = ConceptDatabase(_make_concepts())
        assert len(db.concepts) == 3

    def test_get_concept(self) -> None:
        db = ConceptDatabase(_make_concepts())
        concept = db.get_concept("25064002")
        assert concept is not None
        assert concept.preferred_term == "Headache"

    def test_get_concept_missing(self) -> None:
        db = ConceptDatabase(_make_concepts())
        assert db.get_concept("9999999") is None

    def test_search_by_term_exact(self) -> None:
        db = ConceptDatabase(_make_concepts())
        results = db.search_by_term("headache", exact=True)
        assert "25064002" in results

    def test_search_by_term_exact_synonym(self) -> None:
        db = ConceptDatabase(_make_concepts())
        results = db.search_by_term("cephalalgia", exact=True)
        assert "25064002" in results

    def test_search_by_term_exact_no_match(self) -> None:
        db = ConceptDatabase(_make_concepts())
        results = db.search_by_term("nonexistent", exact=True)
        assert results == []

    def test_search_by_term_prefix(self) -> None:
        db = ConceptDatabase(_make_concepts())
        results = db.search_by_term("head", exact=False)
        assert "25064002" in results

    def test_search_by_term_prefix_case_insensitive(self) -> None:
        db = ConceptDatabase(_make_concepts())
        results = db.search_by_term("HEAD", exact=False)
        assert "25064002" in results

    def test_get_all_terms(self) -> None:
        db = ConceptDatabase(_make_concepts())
        terms = db.get_all_terms("25064002")
        assert "Headache" in terms
        assert "Cephalalgia" in terms
        assert "Head pain" in terms

    def test_get_all_terms_missing(self) -> None:
        db = ConceptDatabase(_make_concepts())
        assert db.get_all_terms("9999999") == []

    def test_get_concepts_by_semantic_tag(self) -> None:
        db = ConceptDatabase(_make_concepts())
        findings = db.get_concepts_by_semantic_tag("finding")
        assert "25064002" in findings
        assert "241615005" not in findings

    def test_set_metadata(self) -> None:
        db = ConceptDatabase()
        db.set_metadata(version="20240101", source="UK Edition")
        assert db.version == "20240101"
        assert db.source == "UK Edition"

    def test_stats(self) -> None:
        db = ConceptDatabase(_make_concepts())
        db.set_metadata(version="v1")
        s = db.stats()
        assert s["total_concepts"] == 3
        assert s["version"] == "v1"
        assert s["concepts_by_semantic_tag"]["finding"] == 1
        assert s["concepts_by_semantic_tag"]["procedure"] == 1

    def test_save_and_load(self, tmp_path) -> None:
        db = ConceptDatabase(_make_concepts())
        db.set_metadata(version="20240101", source="Test")
        path = tmp_path / "concept_db.json"
        db.save(path)

        loaded = ConceptDatabase.load(path)
        assert len(loaded.concepts) == 3
        assert loaded.version == "20240101"
        assert loaded.source == "Test"
        assert loaded.get_concept("25064002") is not None
        assert loaded.get_concept("25064002").preferred_term == "Headache"

    def test_save_unsupported_format(self, tmp_path) -> None:
        import pytest

        db = ConceptDatabase()
        with pytest.raises(ValueError, match="Unsupported format"):
            db.save(tmp_path / "out.xml", output_format="xml")

    def test_build_synonym_index(self) -> None:
        db = ConceptDatabase(_make_concepts())
        # MRI is a synonym of 241615005
        results = db.search_by_term("mri", exact=True)
        assert "241615005" in results
