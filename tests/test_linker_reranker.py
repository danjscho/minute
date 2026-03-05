"""Tests for the cross-encoder reranker.

Uses mocked cross-encoder model for unit tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from worker.snomed.linking.base import LinkedConcept
from worker.snomed.linking.reranker import CrossEncoderReranker, _reranker_cache


def _make_candidates() -> list[LinkedConcept]:
    return [
        LinkedConcept(
            concept_id="25064002",
            preferred_term="Headache",
            fsn="Headache (finding)",
            score=0.95,
            semantic_tag="finding",
        ),
        LinkedConcept(
            concept_id="386661006",
            preferred_term="Fever",
            fsn="Fever (finding)",
            score=0.80,
            semantic_tag="finding",
        ),
        LinkedConcept(
            concept_id="49727002",
            preferred_term="Cough",
            fsn="Cough (finding)",
            score=0.75,
            semantic_tag="finding",
        ),
    ]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the module-level reranker cache between tests."""
    _reranker_cache.clear()
    yield
    _reranker_cache.clear()


class TestCrossEncoderReranker:
    def test_rerank_reorders_by_score(self):
        """Reranker should reorder candidates by cross-encoder scores."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        # Cross-encoder scores: Fever > Cough > Headache (reverses original order)
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.6])
        object.__setattr__(reranker, "_model", mock_model)

        candidates = _make_candidates()
        reranked = reranker.rerank("head pain", candidates)

        assert len(reranked) == 3
        assert reranked[0].concept_id == "386661006"  # Fever (0.9)
        assert reranked[1].concept_id == "49727002"  # Cough (0.6)
        assert reranked[2].concept_id == "25064002"  # Headache (0.3)

    def test_rerank_updates_scores(self):
        """Reranked candidates should have cross-encoder scores, not original."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.85, 0.70, 0.55])
        object.__setattr__(reranker, "_model", mock_model)

        candidates = _make_candidates()
        reranked = reranker.rerank("headache", candidates)

        assert reranked[0].score == 0.85
        assert reranked[1].score == 0.70
        assert reranked[2].score == 0.55

    def test_rerank_with_top_k(self):
        """top_k limits the number of returned candidates."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        object.__setattr__(reranker, "_model", mock_model)

        candidates = _make_candidates()
        reranked = reranker.rerank("headache", candidates, top_k=2)

        assert len(reranked) == 2

    def test_rerank_empty_candidates(self):
        """Empty candidates list returns empty."""
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("headache", [])
        assert reranked == []

    def test_rerank_creates_correct_pairs(self):
        """Cross-encoder should receive (entity_text, preferred_term) pairs."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        object.__setattr__(reranker, "_model", mock_model)

        candidates = _make_candidates()
        reranker.rerank("head pain", candidates)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args == [
            ("head pain", "Headache"),
            ("head pain", "Fever"),
            ("head pain", "Cough"),
        ]

    def test_lazy_loading_not_triggered_on_init(self):
        """Model should not be loaded during __init__."""
        with patch("worker.snomed.linking.reranker.CrossEncoderReranker._load_model") as mock_load:
            CrossEncoderReranker()
            mock_load.assert_not_called()

    def test_model_property_triggers_loading(self):
        """Accessing the model property should trigger model loading."""
        with patch("worker.snomed.linking.reranker.CrossEncoderReranker._load_model") as mock_load:
            reranker = CrossEncoderReranker()
            _ = reranker.model
            mock_load.assert_called_once()

    def test_original_candidates_unchanged(self):
        """Original candidates should not be mutated."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.9, 0.3])
        object.__setattr__(reranker, "_model", mock_model)

        candidates = _make_candidates()
        original_scores = [c.score for c in candidates]
        reranker.rerank("headache", candidates)

        # Original candidates should still have original scores
        for c, orig_score in zip(candidates, original_scores, strict=False):
            assert c.score == orig_score

    def test_custom_model_name(self):
        reranker = CrossEncoderReranker(model_name="custom/reranker")
        assert reranker.model_name == "custom/reranker"
