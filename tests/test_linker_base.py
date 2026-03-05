"""Tests for the entity linking base classes."""

import pytest

from worker.snomed.linking.base import BaseLinkerService, LinkedConcept, LinkResult
from worker.snomed.ner.base import Entity


def _make_entity(text: str = "headache", entity_type: str = "finding") -> Entity:
    return Entity(text=text, start=0, end=len(text), entity_type=entity_type, confidence=0.9)


def _make_linked_concept(concept_id: str = "25064002", score: float = 0.95) -> LinkedConcept:
    return LinkedConcept(
        concept_id=concept_id,
        preferred_term="Headache",
        fsn="Headache (finding)",
        score=score,
        semantic_tag="finding",
    )


class TestLinkedConcept:
    def test_creation(self):
        lc = _make_linked_concept()
        assert lc.concept_id == "25064002"
        assert lc.preferred_term == "Headache"
        assert lc.fsn == "Headache (finding)"
        assert lc.score == 0.95
        assert lc.semantic_tag == "finding"

    def test_equality(self):
        lc1 = _make_linked_concept()
        lc2 = _make_linked_concept()
        assert lc1 == lc2

    def test_inequality_different_score(self):
        lc1 = _make_linked_concept(score=0.9)
        lc2 = _make_linked_concept(score=0.8)
        assert lc1 != lc2


class TestLinkResult:
    def test_with_match(self):
        entity = _make_entity()
        top = _make_linked_concept()
        result = LinkResult(entity=entity, top_concept=top, alternatives=[top])
        assert result.top_concept is not None
        assert result.top_concept.concept_id == "25064002"
        assert len(result.alternatives) == 1

    def test_without_match(self):
        entity = _make_entity()
        result = LinkResult(entity=entity)
        assert result.top_concept is None
        assert result.alternatives == []

    def test_default_alternatives(self):
        entity = _make_entity()
        result = LinkResult(entity=entity, top_concept=_make_linked_concept())
        assert result.alternatives == []


class TestBaseLinkerService:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseLinkerService()

    def test_concrete_subclass(self):
        class ConcreteLinker(BaseLinkerService):
            def link_entity(self, entity: Entity, top_k: int = 5) -> LinkResult:  # noqa: ARG002
                return LinkResult(entity=entity, top_concept=_make_linked_concept())

        linker = ConcreteLinker()
        result = linker.link_entity(_make_entity())
        assert result.top_concept is not None

    def test_batch_default_implementation(self):
        class CountingLinker(BaseLinkerService):
            def __init__(self):
                self.call_count = 0

            def link_entity(self, entity: Entity, top_k: int = 5) -> LinkResult:  # noqa: ARG002
                self.call_count += 1
                return LinkResult(entity=entity, top_concept=_make_linked_concept())

        linker = CountingLinker()
        entities = [_make_entity("headache"), _make_entity("fever"), _make_entity("cough")]
        results = linker.link_entities(entities)
        assert len(results) == 3
        assert linker.call_count == 3

    def test_batch_empty_input(self):
        class SimpleLinker(BaseLinkerService):
            def link_entity(self, entity: Entity, top_k: int = 5) -> LinkResult:  # noqa: ARG002
                return LinkResult(entity=entity)

        linker = SimpleLinker()
        results = linker.link_entities([])
        assert results == []
