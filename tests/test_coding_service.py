"""Tests for the SNOMED coding service orchestrator.

All dependencies (NER service, linker service) are mocked so tests
run without model downloads or SNOMED data.
"""

from unittest.mock import MagicMock
from uuid import uuid4

from common.database.postgres_models import SourceType
from worker.snomed.coding_service import SNOMEDCodingService
from worker.snomed.linking.base import LinkedConcept, LinkResult
from worker.snomed.ner.base import Entity


def _make_entity(text: str = "headache", entity_type: str = "finding") -> Entity:
    return Entity(text=text, start=0, end=len(text), entity_type=entity_type, confidence=0.9)


def _make_link_result(
    entity: Entity,
    concept_id: str = "25064002",
    preferred_term: str = "Headache",
    score: float = 0.95,
) -> LinkResult:
    top_concept = LinkedConcept(
        concept_id=concept_id,
        preferred_term=preferred_term,
        fsn=f"{preferred_term} (finding)",
        score=score,
        semantic_tag="finding",
    )
    return LinkResult(entity=entity, top_concept=top_concept, alternatives=[top_concept])


def _make_link_result_no_match(entity: Entity) -> LinkResult:
    return LinkResult(entity=entity, top_concept=None, alternatives=[])


def _make_service(
    entities: list[Entity] | None = None,
    link_results: list[LinkResult] | None = None,
) -> SNOMEDCodingService:
    """Create a SNOMEDCodingService with mocked NER and linker."""
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = entities or []

    mock_linker = MagicMock()
    mock_linker.link_entities.return_value = link_results or []

    return SNOMEDCodingService(ner_service=mock_ner, linker_service=mock_linker)


class TestSNOMEDCodingService:
    def test_code_text_basic(self):
        """Coding text produces annotations from NER + linking output."""
        entity = _make_entity("headache")
        link_result = _make_link_result(entity)
        service = _make_service(entities=[entity], link_results=[link_result])

        transcription_id = uuid4()
        annotations = service.code_text(
            text="Patient has headache",
            transcription_id=transcription_id,
            source_type=SourceType.TRANSCRIPT,
        )

        assert len(annotations) == 1
        ann = annotations[0]
        assert ann.transcription_id == transcription_id
        assert ann.source_type == SourceType.TRANSCRIPT
        assert ann.text_span == "headache"
        assert ann.snomed_concept_id == "25064002"
        assert ann.snomed_preferred_term == "Headache"
        assert ann.confidence_score == 0.95
        assert ann.entity_type == "finding"

    def test_code_text_calls_ner_then_linker(self):
        """Service calls NER first, then linker with extracted entities."""
        entity = _make_entity()
        link_result = _make_link_result(entity)
        service = _make_service(entities=[entity], link_results=[link_result])

        service.code_text(
            text="Patient has headache",
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        service.ner_service.extract_entities.assert_called_once_with("Patient has headache")
        service.linker_service.link_entities.assert_called_once_with([entity])

    def test_code_text_no_entities(self):
        """When NER finds no entities, no linking is performed."""
        service = _make_service(entities=[], link_results=[])

        annotations = service.code_text(
            text="Normal text with no clinical terms",
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        assert annotations == []
        service.linker_service.link_entities.assert_not_called()

    def test_code_text_no_matches(self):
        """When linker finds no matches, no annotations are created."""
        entity = _make_entity()
        link_result = _make_link_result_no_match(entity)
        service = _make_service(entities=[entity], link_results=[link_result])

        annotations = service.code_text(
            text="Patient has headache",
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        assert annotations == []

    def test_code_text_multiple_entities(self):
        """Multiple entities produce multiple annotations."""
        entity1 = _make_entity("headache", "finding")
        entity2 = Entity(text="fever", start=20, end=25, entity_type="finding", confidence=0.85)
        link1 = _make_link_result(entity1, "25064002", "Headache", 0.95)
        link2 = _make_link_result(entity2, "386661006", "Fever", 0.88)
        service = _make_service(entities=[entity1, entity2], link_results=[link1, link2])

        annotations = service.code_text(
            text="Patient has headache and fever",
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        assert len(annotations) == 2
        assert annotations[0].snomed_concept_id == "25064002"
        assert annotations[1].snomed_concept_id == "386661006"

    def test_code_text_mixed_matches(self):
        """Some entities match, some don't — only matched ones become annotations."""
        entity1 = _make_entity("headache")
        entity2 = _make_entity("unknown term")
        link1 = _make_link_result(entity1)
        link2 = _make_link_result_no_match(entity2)
        service = _make_service(entities=[entity1, entity2], link_results=[link1, link2])

        annotations = service.code_text(
            text="headache and unknown term",
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        assert len(annotations) == 1
        assert annotations[0].snomed_concept_id == "25064002"

    def test_code_text_source_id_propagated(self):
        """source_id is propagated to all annotations."""
        entity = _make_entity()
        link_result = _make_link_result(entity)
        service = _make_service(entities=[entity], link_results=[link_result])

        source_id = uuid4()
        annotations = service.code_text(
            text="headache",
            transcription_id=uuid4(),
            source_type=SourceType.MINUTE_VERSION,
            source_id=source_id,
        )

        assert annotations[0].source_id == source_id
        assert annotations[0].source_type == SourceType.MINUTE_VERSION

    def test_code_text_alternative_concepts(self):
        """Alternative concepts are stored as JSON-serialisable dicts."""
        entity = _make_entity()
        alt1 = LinkedConcept(
            concept_id="25064002", preferred_term="Headache",
            fsn="Headache (finding)", score=0.95, semantic_tag="finding",
        )
        alt2 = LinkedConcept(
            concept_id="386661006", preferred_term="Fever",
            fsn="Fever (finding)", score=0.80, semantic_tag="finding",
        )
        link_result = LinkResult(entity=entity, top_concept=alt1, alternatives=[alt1, alt2])
        service = _make_service(entities=[entity], link_results=[link_result])

        annotations = service.code_text(
            text="headache",
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        assert len(annotations[0].alternative_concepts) == 2
        assert annotations[0].alternative_concepts[0]["concept_id"] == "25064002"
        assert annotations[0].alternative_concepts[1]["concept_id"] == "386661006"

    def test_code_text_unique_ids(self):
        """Each annotation gets a unique ID."""
        entity1 = _make_entity("headache")
        entity2 = _make_entity("fever")
        link1 = _make_link_result(entity1)
        link2 = _make_link_result(entity2, "386661006", "Fever")
        service = _make_service(entities=[entity1, entity2], link_results=[link1, link2])

        annotations = service.code_text(
            text="headache and fever",
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        ids = [a.id for a in annotations]
        assert len(set(ids)) == len(ids)

    def test_lazy_loading_ner(self):
        """NER service is lazy-loaded when not injected."""
        service = SNOMEDCodingService()
        assert service._ner_service is None  # noqa: SLF001

    def test_lazy_loading_linker(self):
        """Linker service is lazy-loaded when not injected."""
        service = SNOMEDCodingService()
        assert service._linker_service is None  # noqa: SLF001
