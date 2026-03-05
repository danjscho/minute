"""
BiomedBERT-based NER service for clinical entity extraction.

Uses a HuggingFace token classification model (default: d4data/biomedical-ner-all)
to extract clinical entities from text. The model is lazy-loaded and cached at
module level following the same pattern as the embedding generator.

Entity types from the NER model (e.g. DISEASE_DISORDER, SIGN_SYMPTOM) are mapped
to SNOMED CT semantic tags (finding, procedure, body_structure) for downstream
entity linking.
"""

from __future__ import annotations

import logging
from typing import Any

from common.settings import get_settings
from worker.snomed.embeddings.generator import _resolve_device
from worker.snomed.ner.base import BaseNERService, Entity
from worker.snomed.ner.chunker import TextChunker

logger = logging.getLogger(__name__)

# Module-level cache for loaded NER pipelines (same pattern as embeddings)
_ner_pipeline_cache: dict[str, Any] = {}

# Mapping from NER model labels to SNOMED CT semantic tags.
# Labels not in this map are preserved as-is (lowercased).
ENTITY_TYPE_MAP: dict[str, str] = {
    # Clinical findings
    "DISEASE_DISORDER": "finding",
    "SIGN_SYMPTOM": "finding",
    # Procedures
    "DIAGNOSTIC_PROCEDURE": "procedure",
    "THERAPEUTIC_PROCEDURE": "procedure",
    # Anatomy
    "BIOLOGICAL_STRUCTURE": "body_structure",
    # Substances
    "MEDICATION": "substance",
    "SUBSTANCE": "substance",
    # Additional useful mappings
    "CLINICAL_EVENT": "finding",
    "LAB_VALUE": "observable_entity",
    "LAB_RESULT": "observable_entity",
}


def _map_entity_type(raw_label: str) -> str:
    """Map a NER model label to a SNOMED CT semantic tag.

    Args:
        raw_label: Raw label from the NER model (e.g. "DISEASE_DISORDER").

    Returns:
        SNOMED semantic tag (e.g. "finding"), or the lowercased raw label
        if no mapping exists.
    """
    return ENTITY_TYPE_MAP.get(raw_label, raw_label.lower())


class BiomedBERTNERService(BaseNERService):
    """NER service using a HuggingFace token classification model.

    Extracts clinical entities from text, handles long documents via
    chunking, and maps entity types to SNOMED semantic tags.

    The model is lazy-loaded on first use and cached at module level
    to avoid reloading across calls and Ray actor restarts.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        confidence_threshold: float | None = None,
        max_seq_length: int | None = None,
    ):
        """Initialise the NER service.

        Args:
            model_name: HuggingFace model ID. Defaults to settings.SNOMED_NER_MODEL.
            device: Device for inference ("auto", "cuda", "cpu", "mps").
            confidence_threshold: Minimum confidence to keep an entity.
            max_seq_length: Maximum token length per chunk.
        """
        settings = get_settings()
        self.model_name = model_name or settings.SNOMED_NER_MODEL
        self.device = _resolve_device(device if device != "auto" else settings.SNOMED_DEVICE)
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else settings.SNOMED_CONFIDENCE_THRESHOLD
        )
        self.max_seq_length = max_seq_length or settings.SNOMED_MAX_SEQ_LENGTH
        self._pipeline = None
        self._chunker = TextChunker(max_tokens=self.max_seq_length)

        logger.info(
            "Initialised BiomedBERT NER service: model=%s, device=%s, threshold=%s",
            self.model_name,
            self.device,
            self.confidence_threshold,
        )

    def _load_model(self) -> None:
        """Lazy load the HuggingFace NER pipeline.

        Uses module-level cache to avoid reloading the model.
        """
        cache_key = f"{self.model_name}:{self.device}"

        if cache_key in _ner_pipeline_cache:
            self._pipeline = _ner_pipeline_cache[cache_key]
            logger.info("Loaded NER pipeline from cache: %s", self.model_name)
            return

        logger.info("Loading NER model: %s on %s", self.model_name, self.device)
        print(f"[NER] Loading model {self.model_name} on {self.device}...")  # noqa: T201

        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)

            device_arg = None if self.device == "cpu" else self.device
            self._pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                device=device_arg,
                aggregation_strategy="simple",
            )

            _ner_pipeline_cache[cache_key] = self._pipeline

            logger.info("Loaded NER model: %s", self.model_name)
            print("[NER] Model loaded successfully")  # noqa: T201

        except Exception as e:
            logger.exception("Failed to load NER model: %s", e)
            print(f"[NER] FAILED to load model: {e}")  # noqa: T201
            raise

    @property
    def pipeline(self) -> Any:
        """Get the loaded NER pipeline (lazy loading)."""
        if self._pipeline is None:
            self._load_model()
        return self._pipeline

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract clinical entities from text.

        Handles long text by chunking, runs NER on each chunk,
        merges overlapping entities, and filters by confidence.

        Args:
            text: Input clinical text.

        Returns:
            List of entities sorted by start position.
        """
        if not text or not text.strip():
            return []

        chunks = self._chunker.chunk_text(text)

        chunk_entities: list[list[Entity]] = []
        for chunk in chunks:
            raw_entities = self._run_ner(chunk.text)
            chunk_entities.append(raw_entities)

        if len(chunks) > 1:
            entities = self._chunker.merge_entities(chunk_entities, chunks)
        elif chunk_entities:
            entities = chunk_entities[0]
        else:
            entities = []

        # Filter by confidence threshold
        entities = [e for e in entities if e.confidence >= self.confidence_threshold]

        logger.info("Extracted %d entities from text (%d chars)", len(entities), len(text))
        return entities

    def _run_ner(self, text: str) -> list[Entity]:
        """Run the NER pipeline on a single text chunk.

        Args:
            text: Text chunk (within model token limit).

        Returns:
            List of Entity objects with chunk-local offsets.
        """
        try:
            raw_results = self.pipeline(text)
        except Exception:
            logger.exception("NER inference failed on text chunk (%d chars)", len(text))
            return []

        entities: list[Entity] = []
        for result in raw_results:
            raw_label = result.get("entity_group", result.get("entity", "UNKNOWN"))
            entity_type = _map_entity_type(raw_label)
            score = float(result.get("score", 0.0))
            start = int(result.get("start", 0))
            end = int(result.get("end", 0))
            word = result.get("word", text[start:end]).strip()

            if not word:
                continue

            entities.append(
                Entity(
                    text=word,
                    start=start,
                    end=end,
                    entity_type=entity_type,
                    confidence=score,
                    raw_label=raw_label,
                )
            )

        return entities

    def extract_entities_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Extract entities from multiple texts.

        Processes each text individually (with chunking support).
        The HuggingFace pipeline handles internal batching.

        Args:
            texts: List of input texts.

        Returns:
            List of entity lists, one per input text.
        """
        return [self.extract_entities(text) for text in texts]
