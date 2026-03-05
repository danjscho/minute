"""SNOMED CT coding service — orchestrates NER and entity linking.

Takes raw clinical text, extracts entities with BiomedBERT NER,
links them to SNOMED CT concepts via SapBERT + FAISS, and returns
SnomedAnnotation database records ready for persistence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from common.database.postgres_models import SnomedAnnotation, SourceType

if TYPE_CHECKING:
    from worker.snomed.linking.base import LinkResult
    from worker.snomed.linking.sapbert_linker import SapBERTLinkerService
    from worker.snomed.ner.biomedbert_ner import BiomedBERTNERService

logger = logging.getLogger(__name__)


class SNOMEDCodingService:
    """Orchestrates NER → entity linking → annotation creation.

    Dependencies are lazy-loaded on first use so the service can be
    constructed cheaply and the heavy model loading happens only when
    coding is actually triggered.
    """

    def __init__(
        self,
        ner_service: BiomedBERTNERService | None = None,
        linker_service: SapBERTLinkerService | None = None,
    ):
        self._ner_service = ner_service
        self._linker_service = linker_service

    @property
    def ner_service(self) -> BiomedBERTNERService:
        if self._ner_service is None:
            from worker.snomed.ner.biomedbert_ner import BiomedBERTNERService

            self._ner_service = BiomedBERTNERService()
        return self._ner_service

    @property
    def linker_service(self) -> SapBERTLinkerService:
        if self._linker_service is None:
            from worker.snomed.linking.sapbert_linker import SapBERTLinkerService

            self._linker_service = SapBERTLinkerService()
        return self._linker_service

    def code_text(
        self,
        text: str,
        transcription_id: UUID,
        source_type: SourceType,
        source_id: UUID | None = None,
    ) -> list[SnomedAnnotation]:
        """Extract clinical entities from text and link to SNOMED CT concepts.

        Args:
            text: Clinical text to process.
            transcription_id: Parent transcription ID for the annotations.
            source_type: Whether the text comes from a transcript, minute, etc.
            source_id: Optional ID of the specific source entity.

        Returns:
            List of SnomedAnnotation records (not yet persisted).
        """
        logger.info(
            "Coding text for transcription %s (source_type=%s, length=%d chars)",
            transcription_id,
            source_type,
            len(text),
        )

        # Step 1: Extract entities
        entities = self.ner_service.extract_entities(text)
        logger.info("NER extracted %d entities", len(entities))

        if not entities:
            return []

        # Step 2: Link entities to SNOMED concepts (batched for efficiency)
        link_results = self.linker_service.link_entities(entities)

        # Step 3: Build annotation records
        annotations = self._build_annotations(
            link_results=link_results,
            transcription_id=transcription_id,
            source_type=source_type,
            source_id=source_id,
        )

        logger.info(
            "Created %d SNOMED annotations from %d entities",
            len(annotations),
            len(entities),
        )
        return annotations

    def _build_annotations(
        self,
        link_results: list[LinkResult],
        transcription_id: UUID,
        source_type: SourceType,
        source_id: UUID | None,
    ) -> list[SnomedAnnotation]:
        """Convert link results to SnomedAnnotation records."""
        annotations: list[SnomedAnnotation] = []

        for result in link_results:
            if result.top_concept is None:
                continue

            annotation = SnomedAnnotation(
                id=uuid4(),
                transcription_id=transcription_id,
                source_type=source_type,
                source_id=source_id,
                text_span=result.entity.text,
                start_char=result.entity.start,
                end_char=result.entity.end,
                entity_type=result.entity.entity_type,
                snomed_concept_id=result.top_concept.concept_id,
                snomed_preferred_term=result.top_concept.preferred_term,
                snomed_fsn=result.top_concept.fsn,
                confidence_score=result.top_concept.score,
                alternative_concepts=[
                    {
                        "concept_id": c.concept_id,
                        "preferred_term": c.preferred_term,
                        "fsn": c.fsn,
                        "confidence_score": c.score,
                    }
                    for c in result.alternatives
                ],
            )
            annotations.append(annotation)

        return annotations
