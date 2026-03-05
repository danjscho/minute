"""
Base classes for entity linking services.

Defines the LinkedConcept and LinkResult dataclasses and the abstract
BaseLinkerService that all linker implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worker.snomed.ner.base import Entity


@dataclass
class LinkedConcept:
    """A SNOMED CT concept matched to an entity by the linker.

    Attributes:
        concept_id: SNOMED CT concept identifier (e.g. "25064002").
        preferred_term: SNOMED preferred term (e.g. "Headache").
        fsn: Fully specified name (e.g. "Headache (finding)").
        score: Similarity score from the linker in [0, 1].
        semantic_tag: SNOMED semantic tag (e.g. "finding", "procedure").
    """

    concept_id: str
    preferred_term: str
    fsn: str
    score: float
    semantic_tag: str


@dataclass
class LinkResult:
    """Result of linking a single NER entity to SNOMED concepts.

    Attributes:
        entity: The original NER entity.
        top_concept: Best matching concept, or None if no match above threshold.
        alternatives: Top-K candidate concepts sorted by score descending.
    """

    entity: Entity
    top_concept: LinkedConcept | None = None
    alternatives: list[LinkedConcept] = field(default_factory=list)


class BaseLinkerService(ABC):
    """Abstract base class for entity linking services.

    All linker implementations must provide link_entity().
    Batch linking has a default sequential implementation
    that subclasses can override for efficiency.
    """

    @abstractmethod
    def link_entity(self, entity: Entity, top_k: int = 5) -> LinkResult:
        """Link a single NER entity to SNOMED CT concepts.

        Args:
            entity: NER entity to link.
            top_k: Number of top candidate concepts to return.

        Returns:
            LinkResult with best match and alternatives.
        """

    def link_entities(self, entities: list[Entity], top_k: int = 5) -> list[LinkResult]:
        """Link multiple NER entities to SNOMED CT concepts.

        Default implementation processes sequentially. Subclasses may
        override for batched embedding and search.

        Args:
            entities: List of NER entities.
            top_k: Number of top candidates per entity.

        Returns:
            List of LinkResult objects, one per entity.
        """
        return [self.link_entity(entity, top_k) for entity in entities]
