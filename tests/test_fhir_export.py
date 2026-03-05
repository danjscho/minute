"""Tests for FHIR R4 export of SNOMED annotations."""

from uuid import uuid4

from common.database.postgres_models import SnomedAnnotation, SourceType
from worker.snomed.export.fhir_export import (
    SNOMED_SYSTEM,
    annotations_to_fhir_bundle,
)


def _make_annotation(**kwargs) -> SnomedAnnotation:
    """Create a test SnomedAnnotation with sensible defaults."""
    defaults = {
        "id": uuid4(),
        "transcription_id": uuid4(),
        "source_type": SourceType.TRANSCRIPT,
        "text_span": "headache",
        "start_char": 0,
        "end_char": 8,
        "entity_type": "finding",
        "snomed_concept_id": "25064002",
        "snomed_preferred_term": "Headache",
        "snomed_fsn": "Headache (finding)",
        "confidence_score": 0.95,
        "is_verified": False,
    }
    defaults.update(kwargs)
    return SnomedAnnotation(**defaults)


class TestFhirExport:
    def test_empty_bundle(self) -> None:
        result = annotations_to_fhir_bundle([], "test-id")
        assert result["resourceType"] == "Bundle"
        assert result["type"] == "collection"

    def test_finding_becomes_condition(self) -> None:
        ann = _make_annotation(entity_type="finding")
        result = annotations_to_fhir_bundle([ann], "test-id")

        assert len(result["entry"]) == 1
        resource = result["entry"][0]["resource"]
        assert resource["resourceType"] == "Condition"
        assert resource["code"]["coding"][0]["system"] == SNOMED_SYSTEM
        assert resource["code"]["coding"][0]["code"] == "25064002"
        assert resource["code"]["coding"][0]["display"] == "Headache"

    def test_disorder_becomes_condition(self) -> None:
        ann = _make_annotation(entity_type="disorder")
        result = annotations_to_fhir_bundle([ann], "test-id")
        assert result["entry"][0]["resource"]["resourceType"] == "Condition"

    def test_procedure_becomes_procedure(self) -> None:
        ann = _make_annotation(
            entity_type="procedure",
            snomed_concept_id="241615005",
            snomed_preferred_term="MRI",
            snomed_fsn="Magnetic resonance imaging (procedure)",
            text_span="MRI",
        )
        result = annotations_to_fhir_bundle([ann], "test-id")

        resource = result["entry"][0]["resource"]
        assert resource["resourceType"] == "Procedure"
        assert resource["status"] == "completed"
        assert resource["code"]["coding"][0]["code"] == "241615005"

    def test_body_structure_becomes_body_structure(self) -> None:
        ann = _make_annotation(
            entity_type="body_structure",
            snomed_concept_id="80891009",
            snomed_preferred_term="Heart",
        )
        result = annotations_to_fhir_bundle([ann], "test-id")

        resource = result["entry"][0]["resource"]
        assert resource["resourceType"] == "BodyStructure"
        assert resource["includedStructure"][0]["structure"]["coding"][0]["code"] == "80891009"

    def test_verified_condition_has_confirmed_status(self) -> None:
        ann = _make_annotation(is_verified=True)
        result = annotations_to_fhir_bundle([ann], "test-id")

        resource = result["entry"][0]["resource"]
        ver_status = resource["verificationStatus"]["coding"][0]["code"]
        assert ver_status == "confirmed"

    def test_unverified_condition_has_unconfirmed_status(self) -> None:
        ann = _make_annotation(is_verified=False)
        result = annotations_to_fhir_bundle([ann], "test-id")

        resource = result["entry"][0]["resource"]
        ver_status = resource["verificationStatus"]["coding"][0]["code"]
        assert ver_status == "unconfirmed"

    def test_multiple_annotations(self) -> None:
        anns = [
            _make_annotation(entity_type="finding"),
            _make_annotation(entity_type="procedure", snomed_concept_id="241615005"),
            _make_annotation(entity_type="body_structure", snomed_concept_id="80891009"),
        ]
        result = annotations_to_fhir_bundle(anns, "test-id")

        assert len(result["entry"]) == 3
        resource_types = [e["resource"]["resourceType"] for e in result["entry"]]
        assert "Condition" in resource_types
        assert "Procedure" in resource_types
        assert "BodyStructure" in resource_types

    def test_bundle_has_identifier(self) -> None:
        result = annotations_to_fhir_bundle([], "my-transcription-id")
        assert result["identifier"]["value"] == "my-transcription-id"

    def test_unknown_entity_type_defaults_to_condition(self) -> None:
        ann = _make_annotation(entity_type="some_unknown_type")
        result = annotations_to_fhir_bundle([ann], "test-id")
        assert result["entry"][0]["resource"]["resourceType"] == "Condition"

    def test_condition_includes_evidence_text(self) -> None:
        ann = _make_annotation(text_span="severe headache")
        result = annotations_to_fhir_bundle([ann], "test-id")
        resource = result["entry"][0]["resource"]
        assert resource["evidence"][0]["concept"]["text"] == "severe headache"

    def test_condition_includes_fsn_note(self) -> None:
        ann = _make_annotation(snomed_fsn="Headache (finding)")
        result = annotations_to_fhir_bundle([ann], "test-id")
        resource = result["entry"][0]["resource"]
        assert resource["note"][0]["text"] == "Headache (finding)"
