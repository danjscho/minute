"""Tests for the SNOMED handler service.

Database access and the coding service are mocked.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from common.database.postgres_models import SnomedAnnotation, SourceType
from common.services.exceptions import SNOMEDCodingFailedError
from common.services.snomed_handler_service import SNOMEDHandlerService


def _make_mock_transcription(dialogue_entries=None):
    """Create a mock transcription object."""
    mock = MagicMock()
    mock.dialogue_entries = dialogue_entries
    return mock


def _make_mock_annotation(transcription_id=None):
    """Create a mock SnomedAnnotation."""
    return SnomedAnnotation(
        id=uuid4(),
        transcription_id=transcription_id or uuid4(),
        source_type=SourceType.TRANSCRIPT,
        text_span="headache",
        start_char=0,
        end_char=8,
        entity_type="finding",
        snomed_concept_id="25064002",
        snomed_preferred_term="Headache",
        snomed_fsn="Headache (finding)",
        confidence_score=0.95,
    )


class TestSNOMEDHandlerService:
    @pytest.mark.asyncio
    async def test_process_skips_when_disabled(self):
        """Coding is skipped when SNOMED_CODING_ENABLED is False."""
        with patch("common.services.snomed_handler_service.get_settings") as mock_settings:
            mock_settings.return_value.SNOMED_CODING_ENABLED = False
            # Should return without error and without touching DB
            await SNOMEDHandlerService.process_snomed_coding(uuid4())

    @pytest.mark.asyncio
    async def test_process_skips_empty_text(self):
        """Coding is skipped when transcription has no text."""
        transcription_id = uuid4()

        with (
            patch("common.services.snomed_handler_service.get_settings") as mock_settings,
            patch.object(SNOMEDHandlerService, "_load_transcription_text", return_value=""),
        ):
            mock_settings.return_value.SNOMED_CODING_ENABLED = True
            await SNOMEDHandlerService.process_snomed_coding(transcription_id)

    @pytest.mark.asyncio
    async def test_process_calls_coding_service(self):
        """Handler loads text, calls coding service, and saves annotations."""
        transcription_id = uuid4()
        mock_annotation = _make_mock_annotation(transcription_id)

        with (
            patch("common.services.snomed_handler_service.get_settings") as mock_settings,
            patch.object(
                SNOMEDHandlerService, "_load_transcription_text",
                return_value="Patient has a headache",
            ),
            patch.object(SNOMEDHandlerService, "_save_annotations") as mock_save,
            patch("worker.snomed.coding_service.SNOMEDCodingService") as mock_cls,
        ):
            mock_settings.return_value.SNOMED_CODING_ENABLED = True
            mock_cls.return_value.code_text.return_value = [mock_annotation]

            await SNOMEDHandlerService.process_snomed_coding(transcription_id)

            mock_cls.return_value.code_text.assert_called_once_with(
                text="Patient has a headache",
                transcription_id=transcription_id,
                source_type=SourceType.TRANSCRIPT,
            )
            mock_save.assert_called_once_with(transcription_id, [mock_annotation], SourceType.TRANSCRIPT)

    @pytest.mark.asyncio
    async def test_process_no_annotations_skips_save(self):
        """When coding produces no annotations, save is not called."""
        transcription_id = uuid4()

        with (
            patch("common.services.snomed_handler_service.get_settings") as mock_settings,
            patch.object(
                SNOMEDHandlerService, "_load_transcription_text",
                return_value="Normal text",
            ),
            patch.object(SNOMEDHandlerService, "_save_annotations") as mock_save,
            patch("worker.snomed.coding_service.SNOMEDCodingService") as mock_cls,
        ):
            mock_settings.return_value.SNOMED_CODING_ENABLED = True
            mock_cls.return_value.code_text.return_value = []

            await SNOMEDHandlerService.process_snomed_coding(transcription_id)
            mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_raises_on_coding_failure(self):
        """Handler raises SNOMEDCodingFailedError on coding failure."""
        transcription_id = uuid4()

        with (
            patch("common.services.snomed_handler_service.get_settings") as mock_settings,
            patch.object(
                SNOMEDHandlerService, "_load_transcription_text",
                return_value="Patient has headache",
            ),
            patch("worker.snomed.coding_service.SNOMEDCodingService") as mock_cls,
        ):
            mock_settings.return_value.SNOMED_CODING_ENABLED = True
            mock_cls.return_value.code_text.side_effect = RuntimeError("model failed")

            with pytest.raises(SNOMEDCodingFailedError):
                await SNOMEDHandlerService.process_snomed_coding(transcription_id)

    @pytest.mark.asyncio
    async def test_process_raises_on_transcription_not_found(self):
        """Handler raises SNOMEDCodingFailedError when transcription not found."""
        with (
            patch("common.services.snomed_handler_service.get_settings") as mock_settings,
            patch.object(
                SNOMEDHandlerService, "_load_transcription_text",
                side_effect=ValueError("Transcription not found"),
            ),
        ):
            mock_settings.return_value.SNOMED_CODING_ENABLED = True

            with pytest.raises(SNOMEDCodingFailedError):
                await SNOMEDHandlerService.process_snomed_coding(uuid4())

    def test_load_transcription_text_concatenates_entries(self):
        """Dialogue entries are concatenated into a single string."""
        transcription_id = uuid4()
        dialogue = [
            {"speaker": "Dr", "text": "Patient has headache", "start_time": 0, "end_time": 5},
            {"speaker": "Nurse", "text": "and fever", "start_time": 5, "end_time": 10},
        ]
        mock_transcription = _make_mock_transcription(dialogue)

        with patch("common.services.snomed_handler_service.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.get.return_value = mock_transcription
            mock_session_cls.return_value = mock_session

            result = SNOMEDHandlerService._load_transcription_text(transcription_id)  # noqa: SLF001

        assert result == "Patient has headache and fever"

    def test_load_transcription_text_empty_entries(self):
        """Empty dialogue entries returns empty string."""
        mock_transcription = _make_mock_transcription(dialogue_entries=None)

        with patch("common.services.snomed_handler_service.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.get.return_value = mock_transcription
            mock_session_cls.return_value = mock_session

            result = SNOMEDHandlerService._load_transcription_text(uuid4())  # noqa: SLF001

        assert result == ""

    def test_load_transcription_text_not_found(self):
        """Raises ValueError when transcription not found."""
        with patch("common.services.snomed_handler_service.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.get.return_value = None
            mock_session_cls.return_value = mock_session

            with pytest.raises(ValueError, match="Transcription not found"):
                SNOMEDHandlerService._load_transcription_text(uuid4())  # noqa: SLF001
