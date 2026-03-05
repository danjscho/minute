"""
FAISS index for SNOMED CT concept similarity search.

Provides efficient nearest neighbor search over concept embeddings
using Facebook AI Similarity Search (FAISS).

Index types:
- Flat: Exact search, good for < 50k vectors
- IVF: Approximate search with inverted file, good for 50k-1M vectors
- IVF+PQ: Compressed approximate search, good for > 1M vectors or memory constraints
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from common.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level cache for loaded indices
_faiss_index_cache: dict[str, "SNOMEDFaissIndex"] = {}

# Threshold for choosing between flat and IVF index types
FLAT_INDEX_THRESHOLD = 10000


class SNOMEDFaissIndex:
    """
    FAISS index for SNOMED CT concept embeddings.

    Supports building, saving, loading, and searching the index.
    The index maps embedding vectors to concept IDs.
    """

    def __init__(self, index_path: str | Path | None = None):
        """
        Initialize FAISS index.

        Args:
            index_path: Path to index file (default from settings)
        """
        settings = get_settings()
        self.index_path = Path(index_path) if index_path else Path(settings.SNOMED_FAISS_INDEX_PATH)
        self._index: Any = None
        self._concept_ids: list[str] = []
        self._embedding_dim: int = 768

    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self._index is not None

    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        return len(self._concept_ids)

    def build(
        self,
        embeddings: np.ndarray,
        concept_ids: list[str],
        index_type: str = "auto",
        nlist: int = 100,
    ) -> None:
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: numpy array of shape (n_concepts, embedding_dim)
            concept_ids: List of concept IDs in same order as embeddings
            index_type: Index type ("flat", "ivf", "auto")
                       "auto" chooses based on number of vectors
            nlist: Number of clusters for IVF index
        """
        import faiss

        n_vectors, embedding_dim = embeddings.shape
        self._embedding_dim = embedding_dim
        self._concept_ids = list(concept_ids)

        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # Choose index type
        if index_type == "auto":
            index_type = "flat" if n_vectors < FLAT_INDEX_THRESHOLD else "ivf"

        logger.info("Building %s index for %d vectors of dim %d", index_type, n_vectors, embedding_dim)
        print(f"[FAISS] Building {index_type} index for {n_vectors} vectors...")  # noqa: T201

        if index_type == "flat":
            # Exact search - good for small datasets
            self._index = faiss.IndexFlatIP(embedding_dim)  # Inner product (for normalized vectors)
            self._index.add(embeddings)

        elif index_type == "ivf":
            # IVF with flat quantizer - approximate search
            # Adjust nlist based on dataset size
            nlist = min(nlist, max(1, n_vectors // 40))

            quantizer = faiss.IndexFlatIP(embedding_dim)
            self._index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train on embeddings
            logger.info("Training IVF index with nlist=%d", nlist)
            self._index.train(embeddings)
            self._index.add(embeddings)

            # Set search parameters
            self._index.nprobe = min(10, nlist)  # Number of clusters to search

        else:
            msg = f"Unknown index type: {index_type}"
            raise ValueError(msg)

        logger.info("Built FAISS index with %d vectors", self._index.ntotal)
        print(f"[FAISS] Index built successfully with {self._index.ntotal} vectors")  # noqa: T201

    def save(self) -> None:
        """
        Save index and concept ID mapping to disk.

        Saves two files:
        - {index_path}: FAISS index binary
        - {index_path}.ids.json: Concept ID mapping
        """
        import faiss

        if self._index is None:
            msg = "No index to save. Call build() first."
            raise ValueError(msg)

        # Create directory if needed
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))
        logger.info("Saved FAISS index to %s", self.index_path)

        # Save concept ID mapping
        ids_path = self.index_path.with_suffix(".ids.json")
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "concept_ids": self._concept_ids,
                    "embedding_dim": self._embedding_dim,
                    "n_vectors": len(self._concept_ids),
                },
                f,
            )
        logger.info("Saved concept IDs to %s", ids_path)
        print(f"[FAISS] Index saved to {self.index_path}")  # noqa: T201

    def load(self) -> None:
        """
        Load index from disk.

        Loads both the FAISS index and concept ID mapping.
        """
        import faiss

        if not self.index_path.exists():
            msg = f"Index file not found: {self.index_path}"
            raise FileNotFoundError(msg)

        ids_path = self.index_path.with_suffix(".ids.json")
        if not ids_path.exists():
            msg = f"Concept IDs file not found: {ids_path}"
            raise FileNotFoundError(msg)

        logger.info("Loading FAISS index from %s", self.index_path)

        # Load FAISS index
        self._index = faiss.read_index(str(self.index_path))

        # Load concept ID mapping
        with ids_path.open(encoding="utf-8") as f:
            data = json.load(f)
            self._concept_ids = data["concept_ids"]
            self._embedding_dim = data.get("embedding_dim", 768)

        logger.info("Loaded FAISS index with %d vectors", self._index.ntotal)
        print(f"[FAISS] Index loaded with {self._index.ntotal} vectors")  # noqa: T201

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, list[list[str]]]:
        """
        Search for nearest neighbors.

        Args:
            query_embeddings: Query vectors of shape (n_queries, embedding_dim)
            top_k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, concept_ids)
            - distances: numpy array of shape (n_queries, top_k)
            - concept_ids: List of lists of concept IDs
        """
        if self._index is None:
            msg = "Index not loaded. Call load() or build() first."
            raise ValueError(msg)

        # Ensure embeddings are float32 and contiguous
        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))

        # Handle single query
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Search
        distances, indices = self._index.search(query_embeddings, top_k)

        # Map indices to concept IDs
        concept_ids_results = []
        for idx_row in indices:
            row_ids = []
            for idx in idx_row:
                if 0 <= idx < len(self._concept_ids):
                    row_ids.append(self._concept_ids[idx])
                else:
                    row_ids.append("")  # Invalid index
            concept_ids_results.append(row_ids)

        return distances, concept_ids_results

    def search_single(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Search for nearest neighbors of a single query.

        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            top_k: Number of nearest neighbors to return

        Returns:
            List of (concept_id, score) tuples sorted by score descending
        """
        distances, concept_ids = self.search(query_embedding, top_k)
        results = []
        for concept_id, score in zip(concept_ids[0], distances[0], strict=False):
            if concept_id:  # Skip empty IDs
                results.append((concept_id, float(score)))
        return results


def get_snomed_faiss_index(index_path: str | Path | None = None) -> SNOMEDFaissIndex:
    """
    Get SNOMED FAISS index with caching.

    Args:
        index_path: Path to index file (default from settings)

    Returns:
        Loaded SNOMEDFaissIndex instance
    """
    settings = get_settings()
    path = str(index_path) if index_path else settings.SNOMED_FAISS_INDEX_PATH

    if path in _faiss_index_cache:
        return _faiss_index_cache[path]

    index = SNOMEDFaissIndex(path)
    index.load()
    _faiss_index_cache[path] = index

    return index
