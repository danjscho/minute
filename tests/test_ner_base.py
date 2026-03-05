"""Tests for the NER base classes and Entity dataclass."""

import pytest

from worker.snomed.ner.base import BaseNERService, Entity


class TestEntity:
    def test_entity_creation(self):
        entity = Entity(
            text="headache",
            start=10,
            end=18,
            entity_type="finding",
            confidence=0.95,
            raw_label="DISEASE_DISORDER",
        )
        assert entity.text == "headache"
        assert entity.start == 10
        assert entity.end == 18
        assert entity.entity_type == "finding"
        assert entity.confidence == 0.95
        assert entity.raw_label == "DISEASE_DISORDER"

    def test_entity_default_raw_label(self):
        entity = Entity(text="fever", start=0, end=5, entity_type="finding", confidence=0.8)
        assert entity.raw_label == ""

    def test_entity_equality(self):
        e1 = Entity(text="fever", start=0, end=5, entity_type="finding", confidence=0.8, raw_label="SIGN_SYMPTOM")
        e2 = Entity(text="fever", start=0, end=5, entity_type="finding", confidence=0.8, raw_label="SIGN_SYMPTOM")
        assert e1 == e2

    def test_entity_inequality_different_span(self):
        e1 = Entity(text="fever", start=0, end=5, entity_type="finding", confidence=0.8)
        e2 = Entity(text="fever", start=10, end=15, entity_type="finding", confidence=0.8)
        assert e1 != e2


class TestBaseNERService:
    def test_cannot_instantiate_abstract(self):
        """BaseNERService is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseNERService()

    def test_concrete_subclass(self):
        """A concrete subclass must implement extract_entities."""

        class ConcreteNER(BaseNERService):
            def extract_entities(self, text: str) -> list[Entity]:  # noqa: ARG002
                return [Entity(text="test", start=0, end=4, entity_type="finding", confidence=1.0)]

        service = ConcreteNER()
        entities = service.extract_entities("test text")
        assert len(entities) == 1
        assert entities[0].text == "test"

    def test_batch_default_implementation(self):
        """Default batch implementation calls extract_entities for each text."""

        class CountingNER(BaseNERService):
            def __init__(self):
                self.call_count = 0

            def extract_entities(self, text: str) -> list[Entity]:
                self.call_count += 1
                return [Entity(text=text, start=0, end=len(text), entity_type="finding", confidence=1.0)]

        service = CountingNER()
        results = service.extract_entities_batch(["text1", "text2", "text3"])
        assert len(results) == 3
        assert service.call_count == 3
        assert results[0][0].text == "text1"
        assert results[2][0].text == "text3"

    def test_batch_empty_input(self):
        class SimpleNER(BaseNERService):
            def extract_entities(self, text: str) -> list[Entity]:  # noqa: ARG002
                return []

        service = SimpleNER()
        results = service.extract_entities_batch([])
        assert results == []
