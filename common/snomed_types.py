"""Shared SNOMED CT data types used by both backend and worker."""

from dataclasses import dataclass, field


@dataclass
class SNOMEDConcept:
    """Represents a SNOMED CT concept with its terms and metadata."""

    concept_id: str
    preferred_term: str = ""
    fsn: str = ""  # Fully Specified Name
    synonyms: list[str] = field(default_factory=list)
    semantic_tag: str = ""  # Extracted from FSN, e.g., "finding", "procedure"
    is_active: bool = True
    parent_ids: list[str] = field(default_factory=list)  # Direct parents via Is-a

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "concept_id": self.concept_id,
            "preferred_term": self.preferred_term,
            "fsn": self.fsn,
            "synonyms": self.synonyms,
            "semantic_tag": self.semantic_tag,
            "is_active": self.is_active,
            "parent_ids": self.parent_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SNOMEDConcept":
        """Create from dictionary."""
        return cls(
            concept_id=data["concept_id"],
            preferred_term=data.get("preferred_term", ""),
            fsn=data.get("fsn", ""),
            synonyms=data.get("synonyms", []),
            semantic_tag=data.get("semantic_tag", ""),
            is_active=data.get("is_active", True),
            parent_ids=data.get("parent_ids", []),
        )
