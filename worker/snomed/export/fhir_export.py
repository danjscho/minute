"""FHIR R4 export for SNOMED CT annotations.

Converts SNOMED annotations to a FHIR R4 Bundle containing
Condition, Procedure, and BodyStructure resources.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fhir.resources.bodystructure import BodyStructure
from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.codeablereference import CodeableReference
from fhir.resources.coding import Coding
from fhir.resources.condition import Condition
from fhir.resources.procedure import Procedure

if TYPE_CHECKING:
    from uuid import UUID

    from common.database.postgres_models import SnomedAnnotation

logger = logging.getLogger(__name__)

SNOMED_SYSTEM = "http://snomed.info/sct"
CONDITION_CLINICAL_STATUS_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-clinical"
CONDITION_VERIFICATION_STATUS_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-ver-status"

# Entity types that map to FHIR Condition resources
CONDITION_ENTITY_TYPES = {"finding", "disorder", "observable_entity", "substance"}
# Entity types that map to FHIR Procedure resources
PROCEDURE_ENTITY_TYPES = {"procedure"}
# Entity types that map to FHIR BodyStructure resources
BODY_STRUCTURE_ENTITY_TYPES = {"body_structure"}


def _make_snomed_coding(annotation: SnomedAnnotation) -> CodeableConcept:
    """Create a SNOMED CT CodeableConcept from an annotation."""
    return CodeableConcept(
        coding=[
            Coding(
                system=SNOMED_SYSTEM,
                code=annotation.snomed_concept_id or "unknown",
                display=annotation.snomed_preferred_term or annotation.text_span,
            )
        ],
        text=annotation.snomed_preferred_term or annotation.text_span,
    )


def _annotation_to_condition(annotation: SnomedAnnotation) -> Condition:
    """Convert a finding/disorder annotation to a FHIR Condition resource."""
    verification_code = "confirmed" if annotation.is_verified else "unconfirmed"

    return Condition(
        subject={"reference": "Patient/unknown"},
        code=_make_snomed_coding(annotation),
        clinicalStatus=CodeableConcept(
            coding=[Coding(system=CONDITION_CLINICAL_STATUS_SYSTEM, code="active")]
        ),
        verificationStatus=CodeableConcept(
            coding=[Coding(system=CONDITION_VERIFICATION_STATUS_SYSTEM, code=verification_code)]
        ),
        note=[{"text": annotation.snomed_fsn}] if annotation.snomed_fsn else None,
        evidence=[
            CodeableReference(
                concept=CodeableConcept(text=annotation.text_span),
            )
        ] if annotation.text_span else None,
    )


def _annotation_to_procedure(annotation: SnomedAnnotation) -> Procedure:
    """Convert a procedure annotation to a FHIR Procedure resource."""
    return Procedure(
        subject={"reference": "Patient/unknown"},
        status="completed",
        code=_make_snomed_coding(annotation),
        note=[{"text": annotation.text_span}] if annotation.text_span else None,
    )


def _annotation_to_body_structure(annotation: SnomedAnnotation) -> BodyStructure:
    """Convert a body_structure annotation to a FHIR BodyStructure resource."""
    return BodyStructure(
        patient={"reference": "Patient/unknown"},
        includedStructure=[{"structure": _make_snomed_coding(annotation)}],
    )


def annotations_to_fhir_bundle(
    annotations: list[SnomedAnnotation],
    transcription_id: str | UUID,
) -> dict:
    """Convert SNOMED annotations to a FHIR R4 Bundle.

    Args:
        annotations: List of SnomedAnnotation database models.
        transcription_id: The source transcription ID (used for Bundle identifier).

    Returns:
        Serializable dict representing a valid FHIR R4 Bundle.
    """
    entries: list[BundleEntry] = []

    for annotation in annotations:
        entity_type = annotation.entity_type or ""

        if entity_type in CONDITION_ENTITY_TYPES:
            resource = _annotation_to_condition(annotation)
        elif entity_type in PROCEDURE_ENTITY_TYPES:
            resource = _annotation_to_procedure(annotation)
        elif entity_type in BODY_STRUCTURE_ENTITY_TYPES:
            resource = _annotation_to_body_structure(annotation)
        else:
            # Default unknown entity types to Condition
            resource = _annotation_to_condition(annotation)

        entries.append(BundleEntry(resource=resource))

    bundle = Bundle(
        type="collection",
        identifier={"value": str(transcription_id)},
        entry=entries if entries else None,
    )

    return bundle.model_dump(exclude_none=True)
