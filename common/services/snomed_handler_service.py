"""Handler service for SNOMED CT coding queue messages.

Follows the same pattern as MinuteHandlerService — classmethods with
SessionLocal() for database access. Called from the Ray worker when
a SNOMED_CODING message is received.
"""

from __future__ import annotations

import logging
import re
from uuid import UUID

from common.database.postgres_database import SessionLocal
from common.database.postgres_models import (
    JobStatus,
    Minute,
    MinuteVersion,
    SnomedAnnotation,
    SourceType,
    Transcription,
)
from common.services.exceptions import SNOMEDCodingFailedError
from common.settings import get_settings

logger = logging.getLogger(__name__)

# Regex to strip HTML tags — avoids importing bs4 into the worker
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    """Convert HTML to plain text by stripping tags."""
    text = _HTML_TAG_RE.sub(" ", html)
    # Collapse whitespace
    return " ".join(text.split())


class SNOMEDHandlerService:
    @classmethod
    async def process_snomed_coding(cls, transcription_id: UUID, source_type: str = "transcript") -> None:
        """Run SNOMED CT coding on transcript or minute text.

        Args:
            transcription_id: ID of the transcription to code.
            source_type: "transcript" or "minute".

        Raises:
            SNOMEDCodingFailedError: If coding fails for any reason.
        """
        settings = get_settings()

        if not settings.SNOMED_CODING_ENABLED:
            logger.info("SNOMED coding is disabled, skipping transcription %s", transcription_id)
            return

        try:
            if source_type == "minute":
                text, db_source_type = cls._load_minute_text(transcription_id), SourceType.MINUTE
            else:
                text, db_source_type = cls._load_transcription_text(transcription_id), SourceType.TRANSCRIPT
        except Exception as e:
            raise SNOMEDCodingFailedError from e

        if not text.strip():
            logger.warning(
                "No text found for transcription %s (source=%s), skipping SNOMED coding",
                transcription_id,
                source_type,
            )
            return

        try:
            from worker.snomed.coding_service import SNOMEDCodingService

            coding_service = SNOMEDCodingService()
            annotations = coding_service.code_text(
                text=text,
                transcription_id=transcription_id,
                source_type=db_source_type,
            )

            if annotations:
                cls._save_annotations(transcription_id, annotations, db_source_type)

            logger.info(
                "SNOMED coding complete for transcription %s (source=%s): %d annotations",
                transcription_id,
                source_type,
                len(annotations),
            )
        except Exception as e:
            logger.exception("SNOMED coding failed for transcription %s", transcription_id)
            raise SNOMEDCodingFailedError from e

    @staticmethod
    def _load_transcription_text(transcription_id: UUID) -> str:
        """Load and concatenate dialogue entries for a transcription."""
        with SessionLocal() as session:
            transcription = session.get(Transcription, transcription_id)
            if not transcription:
                msg = "Transcription not found for id: %s" % transcription_id  # noqa: UP031
                raise ValueError(msg)

            if not transcription.dialogue_entries:
                return ""

            return " ".join(entry["text"] for entry in transcription.dialogue_entries)

    @staticmethod
    def _load_minute_text(transcription_id: UUID) -> str:
        """Load the latest completed minute text for a transcription."""
        with SessionLocal() as session:
            transcription = session.get(Transcription, transcription_id)
            if not transcription:
                msg = "Transcription not found for id: %s" % transcription_id  # noqa: UP031
                raise ValueError(msg)

            # Find the latest minute for this transcription
            minutes = session.query(Minute).filter(
                Minute.transcription_id == transcription_id
            ).all()

            if not minutes:
                logger.warning("No minutes found for transcription %s", transcription_id)
                return ""

            # Get the latest completed minute version across all minutes
            latest_version = None
            for minute in minutes:
                versions = session.query(MinuteVersion).filter(
                    MinuteVersion.minute_id == minute.id,
                    MinuteVersion.status == JobStatus.COMPLETED,
                ).order_by(MinuteVersion.created_datetime.desc()).first()
                if versions and (latest_version is None or versions.created_datetime > latest_version.created_datetime):
                    latest_version = versions

            if not latest_version or not latest_version.html_content:
                logger.warning("No completed minute version found for transcription %s", transcription_id)
                return ""

            return _strip_html(latest_version.html_content)

    @staticmethod
    def _save_annotations(
        transcription_id: UUID, annotations: list[SnomedAnnotation], source_type: SourceType
    ) -> None:
        """Persist SNOMED annotations, replacing existing ones for this transcription and source type."""
        with SessionLocal() as session:
            # Delete existing annotations for this transcription and source type
            existing = session.query(SnomedAnnotation).filter(
                SnomedAnnotation.transcription_id == transcription_id,
                SnomedAnnotation.source_type == source_type,
            ).all()
            for ann in existing:
                session.delete(ann)

            for annotation in annotations:
                session.add(annotation)
            session.commit()

        logger.info(
            "Saved %d annotations for transcription %s (source=%s)",
            len(annotations),
            transcription_id,
            source_type,
        )
