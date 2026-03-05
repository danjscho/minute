"""
Optional cross-encoder reranker for SNOMED concept candidates.

Reranks the top-K candidates from the bi-encoder using a cross-encoder
model that scores entity-concept pairs directly. This can improve accuracy
for difficult cases but adds latency.

Disabled by default — enable via SNOMED_RERANKER_ENABLED setting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from common.settings import get_settings
from worker.snomed.embeddings.generator import _resolve_device

if TYPE_CHECKING:
    from worker.snomed.linking.base import LinkedConcept

logger = logging.getLogger(__name__)

# Module-level cache for loaded cross-encoder models
_reranker_cache: dict[str, Any] = {}


class CrossEncoderReranker:
    """Cross-encoder reranker for SNOMED concept candidates.

    Uses a cross-encoder model to score (entity_text, concept_term) pairs
    and re-sort candidates by cross-encoder score.

    The model is lazy-loaded and cached at module level.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
    ):
        """Initialise the reranker.

        Args:
            model_name: HuggingFace cross-encoder model ID.
            device: Device for inference ("auto", "cuda", "cpu", "mps").
        """
        settings = get_settings()
        self.model_name = model_name
        self.device = _resolve_device(device if device != "auto" else settings.SNOMED_DEVICE)
        self._model = None

        logger.info("Initialised cross-encoder reranker: model=%s, device=%s", self.model_name, self.device)

    def _load_model(self) -> None:
        """Lazy load the cross-encoder model with module-level caching."""
        cache_key = f"{self.model_name}:{self.device}"

        if cache_key in _reranker_cache:
            self._model = _reranker_cache[cache_key]
            logger.info("Loaded reranker from cache: %s", self.model_name)
            return

        logger.info("Loading cross-encoder model: %s on %s", self.model_name, self.device)
        print(f"[Reranker] Loading model {self.model_name} on {self.device}...")  # noqa: T201

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
            _reranker_cache[cache_key] = self._model

            logger.info("Loaded reranker model: %s", self.model_name)
            print("[Reranker] Model loaded successfully")  # noqa: T201

        except Exception as e:
            logger.exception("Failed to load reranker model: %s", e)
            print(f"[Reranker] FAILED to load model: {e}")  # noqa: T201
            raise

    @property
    def model(self) -> Any:
        """Get the loaded cross-encoder model (lazy loading)."""
        if self._model is None:
            self._load_model()
        return self._model

    def rerank(
        self,
        entity_text: str,
        candidates: list[LinkedConcept],
        top_k: int | None = None,
    ) -> list[LinkedConcept]:
        """Rerank candidates using pairwise cross-encoder scoring.

        Creates (entity_text, candidate.preferred_term) pairs, scores
        them with the cross-encoder, and returns candidates sorted by
        cross-encoder score.

        Args:
            entity_text: The original entity text from NER.
            candidates: Bi-encoder candidates to rerank.
            top_k: Maximum number of candidates to return. None returns all.

        Returns:
            Reranked list of LinkedConcept with updated scores.
        """
        from dataclasses import replace

        if not candidates:
            return []

        # Create text pairs for cross-encoder
        pairs = [(entity_text, c.preferred_term) for c in candidates]

        # Score pairs
        scores = self.model.predict(pairs)

        # Build reranked candidates with cross-encoder scores
        reranked = []
        for candidate, score in zip(candidates, scores, strict=False):
            reranked.append(replace(candidate, score=float(score)))

        # Sort by cross-encoder score descending
        reranked.sort(key=lambda c: c.score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked
