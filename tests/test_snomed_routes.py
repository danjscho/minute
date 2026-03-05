"""Tests for the SNOMED API route functions.

Tests the route logic with mocked database sessions and queue services.
Does not use TestClient — tests the async handler functions directly.

The route module creates a queue service at import time (via get_queue_service),
so we mock the queue service module before importing routes.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from common.database.postgres_models import SnomedAnnotation, SourceType

# The backend.api.routes.snomed module calls get_queue_service at module level.
# We need to mock it before the module is imported.
# If the module is already loaded (e.g. from __init__.py), we patch the attribute directly.
_mock_queue = MagicMock()

with patch("common.services.queue_services.get_queue_service", return_value=_mock_queue):
    # Force reimport if already cached
    for mod_name in list(sys.modules):
        if "backend.api.routes.snomed" in mod_name:
            del sys.modules[mod_name]
    if "backend.api.routes" in sys.modules:
        del sys.modules["backend.api.routes"]

    from backend.api.routes.snomed import (
        _annotations_to_csv,
        _verify_transcription_ownership,
        get_snomed_annotations,
        trigger_snomed_coding,
        verify_annotation,
    )


def _make_mock_annotation(transcription_id=None, **overrides):
    """Create a mock SnomedAnnotation with sensible defaults."""
    now = datetime.now(tz=UTC)
    defaults = {
        "id": uuid4(),
        "transcription_id": transcription_id or uuid4(),
        "source_type": SourceType.TRANSCRIPT,
        "source_id": None,
        "text_span": "headache",
        "start_char": 0,
        "end_char": 8,
        "entity_type": "finding",
        "snomed_concept_id": "25064002",
        "snomed_preferred_term": "Headache",
        "snomed_fsn": "Headache (finding)",
        "confidence_score": 0.95,
        "is_verified": False,
        "verified_by": None,
        "alternative_concepts": None,
        "created_datetime": now,
        "updated_datetime": now,
    }
    defaults.update(overrides)
    mock = MagicMock(spec=SnomedAnnotation)
    for key, value in defaults.items():
        setattr(mock, key, value)
    return mock


def _make_mock_transcription(user_id):
    """Create a mock transcription with a user_id."""
    mock = MagicMock()
    mock.user_id = user_id
    return mock


def _make_mock_user(user_id=None):
    """Create a mock user."""
    mock = MagicMock()
    mock.id = user_id or uuid4()
    return mock


class TestVerifyTranscriptionOwnership:
    @pytest.mark.asyncio
    async def test_valid_ownership(self):
        user_id = uuid4()
        transcription_id = uuid4()
        mock_session = AsyncMock()
        mock_transcription = _make_mock_transcription(user_id)
        mock_session.get.return_value = mock_transcription

        result = await _verify_transcription_ownership(mock_session, transcription_id, user_id)
        assert result == mock_transcription

    @pytest.mark.asyncio
    async def test_transcription_not_found(self):
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await _verify_transcription_ownership(mock_session, uuid4(), uuid4())
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_wrong_user(self):
        mock_session = AsyncMock()
        mock_transcription = _make_mock_transcription(uuid4())  # different user
        mock_session.get.return_value = mock_transcription

        with pytest.raises(HTTPException) as exc_info:
            await _verify_transcription_ownership(mock_session, uuid4(), uuid4())
        assert exc_info.value.status_code == 404


class TestGetSnomedAnnotations:
    @pytest.mark.asyncio
    async def test_returns_annotations(self):
        user_id = uuid4()
        transcription_id = uuid4()
        mock_session = AsyncMock()
        mock_user = _make_mock_user(user_id)

        mock_transcription = _make_mock_transcription(user_id)
        mock_session.get.return_value = mock_transcription

        mock_annotation = _make_mock_annotation(transcription_id)
        mock_result = MagicMock()
        mock_result.all.return_value = [mock_annotation]
        mock_session.exec.return_value = mock_result

        result = await get_snomed_annotations(transcription_id, mock_session, mock_user)

        assert result.total_count == 1
        assert len(result.annotations) == 1
        assert result.annotations[0].snomed_concept_id == "25064002"

    @pytest.mark.asyncio
    async def test_returns_empty(self):
        user_id = uuid4()
        mock_session = AsyncMock()
        mock_user = _make_mock_user(user_id)
        mock_session.get.return_value = _make_mock_transcription(user_id)

        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.exec.return_value = mock_result

        result = await get_snomed_annotations(uuid4(), mock_session, mock_user)

        assert result.total_count == 0
        assert result.annotations == []


class TestTriggerSnomedCoding:
    @pytest.mark.asyncio
    async def test_trigger_publishes_message(self):
        user_id = uuid4()
        transcription_id = uuid4()
        mock_session = AsyncMock()
        mock_user = _make_mock_user(user_id)
        mock_session.get.return_value = _make_mock_transcription(user_id)

        with (
            patch("backend.api.routes.snomed.llm_queue_service") as mock_queue,
            patch("backend.api.routes.snomed.settings") as mock_settings,
        ):
            mock_settings.SNOMED_CODING_ENABLED = True
            result = await trigger_snomed_coding(transcription_id, mock_session, mock_user, "transcript")

        assert result["transcription_id"] == str(transcription_id)
        mock_queue.publish_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_disabled_returns_503(self):
        mock_session = AsyncMock()
        mock_user = _make_mock_user()

        with patch("backend.api.routes.snomed.settings") as mock_settings:
            mock_settings.SNOMED_CODING_ENABLED = False
            with pytest.raises(HTTPException) as exc_info:
                await trigger_snomed_coding(uuid4(), mock_session, mock_user)
        assert exc_info.value.status_code == 503


class TestVerifyAnnotation:
    @pytest.mark.asyncio
    async def test_verify_updates_annotation(self):
        user_id = uuid4()
        transcription_id = uuid4()
        annotation_id = uuid4()
        mock_session = AsyncMock()
        mock_user = _make_mock_user(user_id)

        mock_annotation = _make_mock_annotation(transcription_id, id=annotation_id)
        mock_transcription = _make_mock_transcription(user_id)
        mock_session.get.side_effect = [mock_annotation, mock_transcription]

        from common.types import SnomedAnnotationVerifyRequest

        request = SnomedAnnotationVerifyRequest(is_verified=True)

        await verify_annotation(annotation_id, request, mock_session, mock_user)

        assert mock_annotation.is_verified is True
        assert mock_annotation.verified_by == user_id
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_annotation_not_found(self):
        mock_session = AsyncMock()
        mock_user = _make_mock_user()
        mock_session.get.return_value = None

        from common.types import SnomedAnnotationVerifyRequest

        request = SnomedAnnotationVerifyRequest(is_verified=True)

        with pytest.raises(HTTPException) as exc_info:
            await verify_annotation(uuid4(), request, mock_session, mock_user)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_verify_overrides_concept(self):
        user_id = uuid4()
        transcription_id = uuid4()
        mock_session = AsyncMock()
        mock_user = _make_mock_user(user_id)

        mock_annotation = _make_mock_annotation(transcription_id)
        mock_transcription = _make_mock_transcription(user_id)
        mock_session.get.side_effect = [mock_annotation, mock_transcription]

        from common.types import SnomedAnnotationVerifyRequest

        request = SnomedAnnotationVerifyRequest(
            is_verified=True,
            snomed_concept_id="NEW_ID",
            snomed_preferred_term="New Term",
            snomed_fsn="New Term (finding)",
        )

        await verify_annotation(mock_annotation.id, request, mock_session, mock_user)

        assert mock_annotation.snomed_concept_id == "NEW_ID"
        assert mock_annotation.snomed_preferred_term == "New Term"
        assert mock_annotation.snomed_fsn == "New Term (finding)"


class TestAnnotationsToCsv:
    def test_csv_output(self):
        annotation = _make_mock_annotation()
        response = _annotations_to_csv([annotation])
        assert response.media_type == "text/csv"

    def test_csv_empty(self):
        response = _annotations_to_csv([])
        assert response.media_type == "text/csv"
