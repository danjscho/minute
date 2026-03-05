"""Tests for SNOMEDConcept dataclass."""

from common.snomed_types import SNOMEDConcept


class TestSNOMEDConcept:
    def test_creation_with_defaults(self) -> None:
        concept = SNOMEDConcept(concept_id="12345")
        assert concept.concept_id == "12345"
        assert concept.preferred_term == ""
        assert concept.fsn == ""
        assert concept.synonyms == []
        assert concept.semantic_tag == ""
        assert concept.is_active is True
        assert concept.parent_ids == []

    def test_creation_with_all_fields(self) -> None:
        concept = SNOMEDConcept(
            concept_id="25064002",
            preferred_term="Headache",
            fsn="Headache (finding)",
            synonyms=["Cephalalgia", "Head pain"],
            semantic_tag="finding",
            is_active=True,
            parent_ids=["404684003"],
        )
        assert concept.concept_id == "25064002"
        assert concept.preferred_term == "Headache"
        assert len(concept.synonyms) == 2

    def test_to_dict(self) -> None:
        concept = SNOMEDConcept(
            concept_id="25064002",
            preferred_term="Headache",
            fsn="Headache (finding)",
            synonyms=["Cephalalgia"],
            semantic_tag="finding",
            is_active=True,
            parent_ids=["404684003"],
        )
        d = concept.to_dict()
        assert d["concept_id"] == "25064002"
        assert d["preferred_term"] == "Headache"
        assert d["synonyms"] == ["Cephalalgia"]
        assert d["parent_ids"] == ["404684003"]

    def test_from_dict(self) -> None:
        data = {
            "concept_id": "25064002",
            "preferred_term": "Headache",
            "fsn": "Headache (finding)",
            "synonyms": ["Cephalalgia"],
            "semantic_tag": "finding",
            "is_active": True,
            "parent_ids": ["404684003"],
        }
        concept = SNOMEDConcept.from_dict(data)
        assert concept.concept_id == "25064002"
        assert concept.preferred_term == "Headache"

    def test_from_dict_with_missing_optional_fields(self) -> None:
        data = {"concept_id": "12345"}
        concept = SNOMEDConcept.from_dict(data)
        assert concept.concept_id == "12345"
        assert concept.preferred_term == ""
        assert concept.synonyms == []
        assert concept.is_active is True

    def test_round_trip(self) -> None:
        original = SNOMEDConcept(
            concept_id="25064002",
            preferred_term="Headache",
            fsn="Headache (finding)",
            synonyms=["Cephalalgia", "Head pain"],
            semantic_tag="finding",
            is_active=True,
            parent_ids=["404684003", "12345"],
        )
        restored = SNOMEDConcept.from_dict(original.to_dict())
        assert restored.concept_id == original.concept_id
        assert restored.preferred_term == original.preferred_term
        assert restored.fsn == original.fsn
        assert restored.synonyms == original.synonyms
        assert restored.semantic_tag == original.semantic_tag
        assert restored.is_active == original.is_active
        assert restored.parent_ids == original.parent_ids

    def test_synonyms_are_independent_across_instances(self) -> None:
        """Verify dataclass field(default_factory) prevents shared mutable state."""
        a = SNOMEDConcept(concept_id="1")
        b = SNOMEDConcept(concept_id="2")
        a.synonyms.append("test")
        assert b.synonyms == []
