"""
SapBERT embedding generator for SNOMED CT concepts.

Generates dense vector embeddings for concept terms using SapBERT
(Self-Alignment Pretraining for BERT on biomedical entity representations).

SapBERT is specifically designed for biomedical entity linking and produces
embeddings where similar clinical concepts are close in the embedding space.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from common.settings import get_settings

if TYPE_CHECKING:
    from worker.snomed.data.rf2_parser import SNOMEDConcept

logger = logging.getLogger(__name__)

# Module-level cache for loaded models (follow HuggingFace adapter pattern)
_embedding_model_cache: dict[str, tuple[Any, Any]] = {}


def _resolve_device(device: str) -> str:
    """
    Resolve device string to actual device.

    Args:
        device: Device string ("auto", "cuda", "cpu", "mps")

    Returns:
        Resolved device string
    """
    if device == "auto":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


class SNOMEDEmbeddingGenerator:
    """
    Generate embeddings for SNOMED CT concepts using SapBERT.

    SapBERT produces 768-dimensional embeddings optimized for biomedical
    entity linking. The model is lazy-loaded and cached at module level.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 128,
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model ID (default from settings)
            device: Device for inference ("auto", "cuda", "cpu", "mps")
            batch_size: Batch size for embedding generation
            max_length: Maximum token sequence length
        """
        settings = get_settings()
        self.model_name = model_name or settings.SNOMED_LINKER_MODEL
        self.device = _resolve_device(device if device != "auto" else settings.SNOMED_DEVICE)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

        logger.info("Initialized embedding generator with model=%s, device=%s", self.model_name, self.device)

    def _load_model(self) -> None:
        """
        Lazy load the model and tokenizer.

        Models are cached at module level to avoid reloading.
        """
        cache_key = f"{self.model_name}:{self.device}"

        if cache_key in _embedding_model_cache:
            self._model, self._tokenizer = _embedding_model_cache[cache_key]
            logger.info("Loaded embedding model from cache: %s", self.model_name)
            return

        logger.info("Loading embedding model: %s on %s", self.model_name, self.device)
        print(f"[SapBERT] Loading model {self.model_name} on {self.device}...")  # noqa: T201

        try:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()

            # Cache for reuse
            _embedding_model_cache[cache_key] = (self._model, self._tokenizer)

            logger.info("Loaded embedding model: %s", self.model_name)
            print("[SapBERT] Model loaded successfully")  # noqa: T201

        except Exception as e:
            logger.exception("Failed to load embedding model: %s", e)
            print(f"[SapBERT] FAILED to load model: {e}")  # noqa: T201
            raise

    @property
    def model(self) -> Any:
        """Get the loaded model (lazy loading)."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Get the loaded tokenizer (lazy loading)."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (768 for BERT-based models)."""
        return 768

    def embed_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar (requires tqdm)

        Returns:
            numpy array of shape (len(texts), 768)
        """
        import torch

        if not texts:
            return np.array([])

        all_embeddings = []
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        if show_progress:
            try:
                from tqdm import tqdm

                batches = tqdm(batches, desc="Generating embeddings")
            except ImportError:
                pass

        with torch.no_grad():
            for batch_texts in batches:
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(**inputs)

                # Use CLS token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :]

                # Normalize embeddings (important for similarity search)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (768,)
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def _collect_concept_terms(self, concept: SNOMEDConcept, concept_id: str) -> list[str]:
        """Collect all unique terms for a concept (preferred term, FSN, synonyms)."""
        terms: list[str] = []
        if concept.preferred_term:
            terms.append(concept.preferred_term)
        if concept.fsn:
            fsn_term = concept.fsn.rsplit(" (", 1)[0] if "(" in concept.fsn else concept.fsn
            if fsn_term not in terms:
                terms.append(fsn_term)
        for syn in concept.synonyms:
            if syn not in terms:
                terms.append(syn)
        return terms or [concept_id]

    def _embed_concept(self, concept: SNOMEDConcept, concept_id: str, use_all_terms: bool) -> np.ndarray:
        """Generate embedding for a single concept."""
        if use_all_terms:
            terms = self._collect_concept_terms(concept, concept_id)
            term_embeddings = self.embed_texts(terms)
            concept_embedding = np.mean(term_embeddings, axis=0)
            return concept_embedding / np.linalg.norm(concept_embedding)
        term = concept.preferred_term or concept.fsn or concept_id
        return self.embed_single(term)

    def embed_concepts(
        self,
        concepts: dict[str, Any],
        use_all_terms: bool = True,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Generate embeddings for SNOMED CT concepts.

        For each concept, generates embeddings for all terms (preferred term,
        FSN, synonyms) and averages them to get a single concept embedding.

        Args:
            concepts: Dictionary mapping concept_id to SNOMEDConcept
            use_all_terms: If True, average embeddings of all terms per concept.
                          If False, only use preferred term.
            show_progress: Show progress bar

        Returns:
            Tuple of (embeddings array, list of concept IDs in same order)
        """
        concept_ids = list(concepts.keys())
        concept_embeddings = []

        if show_progress:
            try:
                from tqdm import tqdm

                concept_ids_iter = tqdm(concept_ids, desc="Embedding concepts")
            except ImportError:
                concept_ids_iter = concept_ids
        else:
            concept_ids_iter = concept_ids

        for concept_id in concept_ids_iter:
            concept: SNOMEDConcept = concepts[concept_id]
            concept_embeddings.append(self._embed_concept(concept, concept_id, use_all_terms))

        return np.vstack(concept_embeddings), list(concepts.keys())
