"""Tests for the BiomedBERT NER service.

Uses mocked HuggingFace pipeline for unit tests. Integration tests
requiring actual model downloads should be marked with @pytest.mark.slow.
"""

from unittest.mock import MagicMock, patch

import pytest

from worker.snomed.ner.biomedbert_ner import (
    ENTITY_TYPE_MAP,
    BiomedBERTNERService,
    _map_entity_type,
    _ner_pipeline_cache,
)


class TestEntityTypeMapping:
    def test_disease_disorder_maps_to_finding(self):
        assert _map_entity_type("DISEASE_DISORDER") == "finding"

    def test_sign_symptom_maps_to_finding(self):
        assert _map_entity_type("SIGN_SYMPTOM") == "finding"

    def test_diagnostic_procedure_maps_to_procedure(self):
        assert _map_entity_type("DIAGNOSTIC_PROCEDURE") == "procedure"

    def test_therapeutic_procedure_maps_to_procedure(self):
        assert _map_entity_type("THERAPEUTIC_PROCEDURE") == "procedure"

    def test_biological_structure_maps_to_body_structure(self):
        assert _map_entity_type("BIOLOGICAL_STRUCTURE") == "body_structure"

    def test_medication_maps_to_substance(self):
        assert _map_entity_type("MEDICATION") == "substance"

    def test_unknown_label_lowercased(self):
        assert _map_entity_type("SOME_NEW_TYPE") == "some_new_type"

    def test_all_mapped_types_have_valid_targets(self):
        valid_targets = {"finding", "procedure", "body_structure", "substance", "observable_entity"}
        for target in ENTITY_TYPE_MAP.values():
            assert target in valid_targets, f"Unexpected mapping target: {target}"


def _mock_pipeline_result():
    """Return mock HuggingFace NER pipeline output."""
    return [
        {
            "entity_group": "DISEASE_DISORDER",
            "score": 0.95,
            "word": "headache",
            "start": 22,
            "end": 30,
        },
        {
            "entity_group": "SIGN_SYMPTOM",
            "score": 0.88,
            "word": "fever",
            "start": 35,
            "end": 40,
        },
        {
            "entity_group": "BIOLOGICAL_STRUCTURE",
            "score": 0.45,
            "word": "blood",
            "start": 42,
            "end": 47,
        },
    ]


def _make_service_with_mock_pipeline(mock_results, **kwargs):
    """Create a BiomedBERTNERService with a mocked pipeline (bypasses model loading)."""
    service = BiomedBERTNERService(**kwargs)
    mock_pipeline = MagicMock()
    if isinstance(mock_results, Exception):
        mock_pipeline.side_effect = mock_results
    else:
        mock_pipeline.return_value = mock_results
    # Bypass lazy loading by setting the pipeline directly
    object.__setattr__(service, "_pipeline", mock_pipeline)
    return service, mock_pipeline


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the module-level pipeline cache between tests."""
    _ner_pipeline_cache.clear()
    yield
    _ner_pipeline_cache.clear()


class TestBiomedBERTNERService:
    def test_extract_entities_basic(self):
        """Basic entity extraction with mocked pipeline."""
        service, _ = _make_service_with_mock_pipeline(_mock_pipeline_result(), confidence_threshold=0.5)

        text = "Patient presents with headache and fever. Blood pressure elevated."
        entities = service.extract_entities(text)

        assert len(entities) == 2  # headache + fever (blood at 0.45 below 0.5 threshold)
        assert entities[0].text == "headache"
        assert entities[0].entity_type == "finding"
        assert entities[0].confidence == 0.95
        assert entities[1].text == "fever"
        assert entities[1].entity_type == "finding"

    def test_confidence_filtering(self):
        """Entities below confidence threshold should be filtered."""
        service, _ = _make_service_with_mock_pipeline(_mock_pipeline_result(), confidence_threshold=0.9)

        entities = service.extract_entities("test text")
        # Only headache (0.95) passes the 0.9 threshold
        assert len(entities) == 1
        assert entities[0].text == "headache"

    def test_empty_text_returns_empty(self):
        service, _ = _make_service_with_mock_pipeline([])
        assert service.extract_entities("") == []
        assert service.extract_entities("   ") == []

    def test_no_entities_found(self):
        service, _ = _make_service_with_mock_pipeline([], confidence_threshold=0.5)
        entities = service.extract_entities("Nothing clinical here.")
        assert entities == []

    def test_entity_offsets_preserved(self):
        """Character offsets from the pipeline should be preserved."""
        results = [{"entity_group": "DISEASE_DISORDER", "score": 0.9, "word": "headache", "start": 22, "end": 30}]
        service, _ = _make_service_with_mock_pipeline(results, confidence_threshold=0.5)

        entities = service.extract_entities("Patient presents with headache.")
        assert entities[0].start == 22
        assert entities[0].end == 30

    def test_raw_label_preserved(self):
        """Original NER label should be preserved in raw_label."""
        results = [{"entity_group": "THERAPEUTIC_PROCEDURE", "score": 0.9, "word": "surgery", "start": 0, "end": 7}]
        service, _ = _make_service_with_mock_pipeline(results, confidence_threshold=0.5)

        entities = service.extract_entities("surgery")
        assert entities[0].raw_label == "THERAPEUTIC_PROCEDURE"
        assert entities[0].entity_type == "procedure"

    def test_batch_extraction(self):
        """Batch extraction should process each text."""
        results = [{"entity_group": "DISEASE_DISORDER", "score": 0.9, "word": "test", "start": 0, "end": 4}]
        service, mock_pipeline = _make_service_with_mock_pipeline(results, confidence_threshold=0.5)

        results = service.extract_entities_batch(["text1", "text2"])
        assert len(results) == 2
        assert mock_pipeline.call_count == 2

    def test_pipeline_error_returns_empty(self):
        """If the pipeline raises, _run_ner should return empty list."""
        service, _ = _make_service_with_mock_pipeline(RuntimeError("Model error"), confidence_threshold=0.5)

        entities = service.extract_entities("some text")
        assert entities == []

    def test_lazy_loading_not_triggered_on_init(self):
        """Model should not be loaded during __init__."""
        with patch("worker.snomed.ner.biomedbert_ner.BiomedBERTNERService._load_model") as mock_load:
            BiomedBERTNERService()
            mock_load.assert_not_called()

    def test_pipeline_property_triggers_loading(self):
        """Accessing the pipeline property should trigger model loading."""
        with patch("worker.snomed.ner.biomedbert_ner.BiomedBERTNERService._load_model") as mock_load:
            service = BiomedBERTNERService()
            _ = service.pipeline
            mock_load.assert_called_once()

    def test_uses_settings_defaults(self):
        """Service should use settings values when not overridden."""
        service = BiomedBERTNERService()
        assert service.model_name == "d4data/biomedical-ner-all"

    def test_custom_model_name(self):
        service = BiomedBERTNERService(model_name="custom/model")
        assert service.model_name == "custom/model"

    def test_custom_confidence_threshold(self):
        service = BiomedBERTNERService(confidence_threshold=0.3)
        assert service.confidence_threshold == 0.3
