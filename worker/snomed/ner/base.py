"""
Base classes for Named Entity Recognition (NER) services.

Defines the Entity dataclass and abstract BaseNERService that all
NER implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Entity:
    """A clinical entity extracted from text by an NER model.

    Attributes:
        text: The matched text span from the source.
        start: Character offset of the entity start in the source text.
        end: Character offset of the entity end in the source text.
        entity_type: SNOMED semantic tag (e.g. "finding", "procedure", "body_structure").
        confidence: Model confidence score in [0, 1].
        raw_label: Original model label before mapping (e.g. "DISEASE_DISORDER").
    """

    text: str
    start: int
    end: int
    entity_type: str
    confidence: float
    raw_label: str = ""


class BaseNERService(ABC):
    """Abstract base class for NER services.

    All NER implementations must provide extract_entities().
    Batch extraction has a default sequential implementation
    that subclasses can override for efficiency.
    """

    @abstractmethod
    def extract_entities(self, text: str) -> list[Entity]:
        """Extract clinical entities from text.

        Args:
            text: Input text to analyse for clinical entities.

        Returns:
            List of extracted entities with character offsets and types.
        """

    def extract_entities_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Extract entities from multiple texts.

        Default implementation processes sequentially. Subclasses may
        override for batched inference.

        Args:
            texts: List of input texts.

        Returns:
            List of entity lists, one per input text.
        """
        return [self.extract_entities(text) for text in texts]
