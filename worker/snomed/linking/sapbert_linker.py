"""
SapBERT bi-encoder linker for mapping NER entities to SNOMED CT concepts.

Embeds entity text with SapBERT, searches a FAISS index for nearest
neighbours, and returns ranked SNOMED concept candidates with metadata
from the concept database.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from common.settings import get_settings
from worker.snomed.linking.base import BaseLinkerService, LinkedConcept, LinkResult

if TYPE_CHECKING:
    from worker.snomed.data.concept_db import ConceptDatabase
    from worker.snomed.embeddings.faiss_index import SNOMEDFaissIndex
    from worker.snomed.embeddings.generator import SNOMEDEmbeddingGenerator
    from worker.snomed.ner.base import Entity

logger = logging.getLogger(__name__)

# Score boost for candidates whose semantic tag matches the NER entity type.
# This is a soft preference — mismatched types are not excluded, just ranked lower.
SEMANTIC_TAG_BOOST = 0.05


class SapBERTLinkerService(BaseLinkerService):
    """Entity linker using SapBERT embeddings and FAISS similarity search.

    Connects three Phase 2 components:
    - SNOMEDEmbeddingGenerator: embeds entity text
    - SNOMEDFaissIndex: finds nearest concept embeddings
    - ConceptDatabase: retrieves concept metadata

    Dependencies can be injected for testing or loaded from settings.
    """

    def __init__(
        self,
        concept_db: ConceptDatabase | None = None,
        embedding_generator: SNOMEDEmbeddingGenerator | None = None,
        faiss_index: SNOMEDFaissIndex | None = None,
        confidence_threshold: float | None = None,
        top_k: int | None = None,
        filter_by_semantic_tag: bool = True,
    ):
        """Initialise the SapBERT linker.

        Args:
            concept_db: Concept database for metadata lookup. Loaded from settings if None.
            embedding_generator: Embedding generator. Created from settings if None.
            faiss_index: FAISS index. Loaded from settings if None.
            confidence_threshold: Minimum score to accept a match.
            top_k: Default number of top candidates to return.
            filter_by_semantic_tag: Whether to boost candidates matching entity type.
        """
        settings = get_settings()

        self._concept_db = concept_db
        self._embedding_generator = embedding_generator
        self._faiss_index = faiss_index
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else settings.SNOMED_CONFIDENCE_THRESHOLD
        )
        self.default_top_k = top_k or settings.SNOMED_TOP_K_CANDIDATES
        self.filter_by_semantic_tag = filter_by_semantic_tag

        logger.info(
            "Initialised SapBERT linker: threshold=%s, top_k=%s, semantic_filter=%s",
            self.confidence_threshold,
            self.default_top_k,
            self.filter_by_semantic_tag,
        )

    @property
    def concept_db(self) -> ConceptDatabase:
        """Get concept database (lazy loaded from settings if not injected)."""
        if self._concept_db is None:
            from pathlib import Path

            from worker.snomed.data.concept_db import ConceptDatabase

            settings = get_settings()
            db_path = Path(settings.SNOMED_CT_DATA_PATH) / "concept_db.json"
            self._concept_db = ConceptDatabase.load(db_path)
        return self._concept_db

    @property
    def embedding_generator(self) -> SNOMEDEmbeddingGenerator:
        """Get embedding generator (lazy created if not injected)."""
        if self._embedding_generator is None:
            from worker.snomed.embeddings.generator import SNOMEDEmbeddingGenerator

            self._embedding_generator = SNOMEDEmbeddingGenerator()
        return self._embedding_generator

    @property
    def faiss_index(self) -> SNOMEDFaissIndex:
        """Get FAISS index (lazy loaded from settings if not injected)."""
        if self._faiss_index is None:
            from worker.snomed.embeddings.faiss_index import get_snomed_faiss_index

            self._faiss_index = get_snomed_faiss_index()
        return self._faiss_index

    def link_entity(self, entity: Entity, top_k: int | None = None) -> LinkResult:
        """Link a single NER entity to SNOMED CT concepts.

        Args:
            entity: NER entity to link.
            top_k: Number of top candidates. Uses default_top_k if None.

        Returns:
            LinkResult with best match and alternatives.
        """
        top_k = top_k or self.default_top_k

        # Embed entity text
        embedding = self.embedding_generator.embed_single(entity.text)

        # Search FAISS index
        search_results = self.faiss_index.search_single(embedding, top_k=top_k)

        # Build linked concepts with metadata
        candidates = self._build_candidates(search_results, entity)

        # Filter by confidence threshold
        candidates = [c for c in candidates if c.score >= self.confidence_threshold]

        top_concept = candidates[0] if candidates else None
        return LinkResult(entity=entity, top_concept=top_concept, alternatives=candidates)

    def link_entities(self, entities: list[Entity], top_k: int | None = None) -> list[LinkResult]:
        """Link multiple NER entities with batched embedding and search.

        More efficient than sequential link_entity() for multiple entities
        because embeddings are generated in a single batch.

        Args:
            entities: List of NER entities.
            top_k: Number of top candidates per entity.

        Returns:
            List of LinkResult objects, one per entity.
        """
        if not entities:
            return []

        top_k = top_k or self.default_top_k

        # Batch embed all entity texts
        texts = [e.text for e in entities]
        embeddings = self.embedding_generator.embed_texts(texts)

        # Batch FAISS search
        distances, concept_ids_batch = self.faiss_index.search(embeddings, top_k=top_k)

        # Build results
        results: list[LinkResult] = []
        for i, entity in enumerate(entities):
            search_results = [
                (cid, float(score))
                for cid, score in zip(concept_ids_batch[i], distances[i], strict=False)
                if cid  # Skip empty IDs
            ]

            candidates = self._build_candidates(search_results, entity)
            candidates = [c for c in candidates if c.score >= self.confidence_threshold]

            top_concept = candidates[0] if candidates else None
            results.append(LinkResult(entity=entity, top_concept=top_concept, alternatives=candidates))

        return results

    def _build_candidates(
        self,
        search_results: list[tuple[str, float]],
        entity: Entity,
    ) -> list[LinkedConcept]:
        """Build LinkedConcept objects from FAISS search results.

        Looks up concept metadata from the concept database and optionally
        applies semantic tag boosting.

        Args:
            search_results: List of (concept_id, score) from FAISS search.
            entity: Original NER entity (used for semantic tag matching).

        Returns:
            List of LinkedConcept sorted by score descending.
        """
        candidates: list[LinkedConcept] = []

        for concept_id, score in search_results:
            concept = self.concept_db.get_concept(concept_id)
            if concept is None:
                logger.warning("Concept %s not found in database, skipping", concept_id)
                continue

            # Apply semantic tag boost/penalty
            adjusted_score = score
            if self.filter_by_semantic_tag and entity.entity_type:
                if concept.semantic_tag == entity.entity_type:
                    adjusted_score = min(1.0, score + SEMANTIC_TAG_BOOST)
                else:
                    adjusted_score = max(0.0, score - SEMANTIC_TAG_BOOST)

            candidates.append(
                LinkedConcept(
                    concept_id=concept_id,
                    preferred_term=concept.preferred_term,
                    fsn=concept.fsn,
                    score=adjusted_score,
                    semantic_tag=concept.semantic_tag,
                )
            )

        # Re-sort after score adjustment
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
