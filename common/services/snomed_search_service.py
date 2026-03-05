"""Lightweight SNOMED concept search service for the backend API.

Loads the concept_db.json file and provides text-based search via a synonym index.
No ML models or GPU required — purely CPU/RAM based.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from common.snomed_types import SNOMEDConcept

logger = logging.getLogger(__name__)


class SnomedConceptSearchService:
    """Singleton search service that loads ConceptDatabase JSON and provides prefix search."""

    _instance: SnomedConceptSearchService | None = None

    def __init__(self) -> None:
        self.concepts: dict[str, SNOMEDConcept] = {}
        self._synonym_index: dict[str, set[str]] = {}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> SnomedConceptSearchService:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, data_path: str) -> None:
        """Load concept database from JSON file. No-op if already loaded."""
        if self._loaded:
            return

        concept_db_path = Path(data_path) / "concept_db.json"
        if not concept_db_path.exists():
            logger.warning("SNOMED concept_db.json not found at %s", concept_db_path)
            return

        with concept_db_path.open(encoding="utf-8") as f:
            data = json.load(f)

        self.concepts = {cid: SNOMEDConcept.from_dict(cdata) for cid, cdata in data.get("concepts", {}).items()}
        self._build_synonym_index()
        self._loaded = True
        logger.info("Loaded %d SNOMED concepts for search", len(self.concepts))

    def _build_synonym_index(self) -> None:
        """Build index mapping normalized terms to concept IDs."""
        self._synonym_index.clear()

        for concept_id, concept in self.concepts.items():
            terms_to_index: list[str] = []

            if concept.preferred_term:
                terms_to_index.append(concept.preferred_term)

            if concept.fsn:
                fsn_term = concept.fsn.rsplit(" (", 1)[0] if "(" in concept.fsn else concept.fsn
                terms_to_index.append(fsn_term)

            terms_to_index.extend(concept.synonyms)

            for term in terms_to_index:
                normalized = term.lower().strip()
                if normalized:
                    if normalized not in self._synonym_index:
                        self._synonym_index[normalized] = set()
                    self._synonym_index[normalized].add(concept_id)

        logger.info("Built synonym index with %d unique terms", len(self._synonym_index))

    def search(self, query: str, limit: int = 10) -> list[SNOMEDConcept]:
        """Search concepts by prefix match on synonym index.

        Results are sorted with exact preferred_term matches first,
        then alphabetically by preferred_term.
        """
        normalized = query.lower().strip()
        if not normalized:
            return []

        matching_ids: set[str] = set()

        # Exact match first
        if normalized in self._synonym_index:
            matching_ids.update(self._synonym_index[normalized])

        # Prefix match to fill remaining results
        if len(matching_ids) < limit * 3:
            for term, concept_ids in self._synonym_index.items():
                if term.startswith(normalized) and term != normalized:
                    matching_ids.update(concept_ids)
                    if len(matching_ids) >= limit * 3:
                        break

        results = [self.concepts[cid] for cid in matching_ids if cid in self.concepts]
        results.sort(
            key=lambda c: (
                0 if c.preferred_term.lower() == normalized else 1,
                c.preferred_term.lower(),
            )
        )
        return results[:limit]

    def get_concept(self, concept_id: str) -> SNOMEDConcept | None:
        """Get a concept by ID."""
        return self.concepts.get(concept_id)
