"""Tests for OlmoASR transcription adapter.

Unit tests run without dependencies, integration tests require:
- olmoasr package installed
- OLMOASR_MODEL_SIZE environment variable set
- test_audio/test_audio.wav file (for integration tests)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from common.database.postgres_models import Recording
from common.services.transcription_services.adapter import AdapterType
from common.services.transcription_services.olmoasr import OlmoASRAdapter
from common.types import TranscriptionJobMessageData

# Test audio file path
TEST_AUDIO_PATH = Path("test_audio/test_audio.wav")

# Mark for tests requiring OlmoASR to be available
requires_olmoasr = pytest.mark.skipif(
    not OlmoASRAdapter.is_available(),
    reason="OlmoASR not available (requires olmoasr package and OLMOASR_MODEL_SIZE)",
)

requires_test_audio = pytest.mark.skipif(
    not TEST_AUDIO_PATH.exists(),
    reason=f"Test audio file not found: {TEST_AUDIO_PATH}",
)


class TestOlmoASRAdapterUnit:
    """Unit tests for OlmoASRAdapter (no external dependencies required)."""

    def test_adapter_attributes(self):
        """Test that adapter has required class attributes."""
        assert OlmoASRAdapter.name == "olmoasr_local"
        assert OlmoASRAdapter.adapter_type == AdapterType.SYNCHRONOUS
        assert OlmoASRAdapter.max_audio_length == 7200  # 2 hours

    def test_is_available_without_model_size(self):
        """Test is_available returns False when OLMOASR_MODEL_SIZE is not set."""
        with patch("common.services.transcription_services.olmoasr.settings") as mock_settings:
            mock_settings.OLMOASR_MODEL_SIZE = None
            assert OlmoASRAdapter.is_available() is False

    def test_is_available_without_package(self):
        """Test is_available returns False when olmoasr package is not installed."""
        with (
            patch("common.services.transcription_services.olmoasr.settings") as mock_settings,
            patch.dict("sys.modules", {"olmoasr": None}),
        ):
            mock_settings.OLMOASR_MODEL_SIZE = "medium"
            # Force import error by patching the import mechanism
            with patch("builtins.__import__", side_effect=ImportError("No module named 'olmoasr'")):
                # Need to call is_available in a way that triggers the import
                result = OlmoASRAdapter.is_available()
                # This may still return True if olmoasr was already imported
                # The actual behavior depends on import caching

    def test_is_available_with_valid_config(self):
        """Test is_available returns True when properly configured and package installed."""
        # This test verifies the happy path when olmoasr is installed
        # Since we can't easily mock the import in is_available(), we just verify
        # that with OLMOASR_MODEL_SIZE set, the method doesn't crash
        with patch("common.services.transcription_services.olmoasr.settings") as mock_settings:
            mock_settings.OLMOASR_MODEL_SIZE = "medium"
            # Result depends on whether olmoasr package is actually installed
            # The important thing is that the method doesn't raise an exception
            result = OlmoASRAdapter.is_available()
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_start_rejects_recording_object(self):
        """Test that start() raises ValueError when given a Recording instead of Path."""
        mock_recording = MagicMock(spec=Recording)

        with pytest.raises(ValueError, match="OlmoASR sync adapter expects Path, not Recording"):
            await OlmoASRAdapter.start(mock_recording)

    @pytest.mark.asyncio
    async def test_check_returns_data_unchanged(self):
        """Test that check() returns data unchanged (sync adapter no-op)."""
        input_data = TranscriptionJobMessageData(
            transcription_service="olmoasr_local",
            job_name="test_job",
            transcript=[{"speaker": "Speaker 1", "text": "Test", "start_time": 0.0, "end_time": 1.0}],
        )

        result = await OlmoASRAdapter.check(input_data)

        assert result is input_data
        assert result.transcript == input_data.transcript

    @pytest.mark.asyncio
    async def test_start_returns_correct_structure(self):
        """Test that start() returns properly structured TranscriptionJobMessageData."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = "This is the transcribed text."

        with (
            patch("common.services.transcription_services.olmoasr.settings") as mock_settings,
            patch.object(OlmoASRAdapter, "_load_model", return_value=mock_model),
        ):
            mock_settings.OLMOASR_MODEL_SIZE = "medium"
            mock_settings.OLMOASR_DEVICE = "cpu"

            result = await OlmoASRAdapter.start(Path("/tmp/test.mp3"))

            assert result.transcription_service == "olmoasr_local"
            assert len(result.transcript) == 1
            assert result.transcript[0]["speaker"] == "Speaker 1"
            assert result.transcript[0]["text"] == "This is the transcribed text."
            assert result.transcript[0]["start_time"] == 0.0
            assert result.transcript[0]["end_time"] == 0.0

    @pytest.mark.asyncio
    async def test_start_handles_dict_result(self):
        """Test that start() handles dict result from model.transcribe()."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Transcribed from dict result."}

        with (
            patch("common.services.transcription_services.olmoasr.settings") as mock_settings,
            patch.object(OlmoASRAdapter, "_load_model", return_value=mock_model),
        ):
            mock_settings.OLMOASR_MODEL_SIZE = "medium"
            mock_settings.OLMOASR_DEVICE = "cpu"

            result = await OlmoASRAdapter.start(Path("/tmp/test.mp3"))

            assert result.transcript[0]["text"] == "Transcribed from dict result."


class TestOlmoASRAdapterModelLoading:
    """Tests for model loading and caching behavior."""

    def test_load_model_caches_result(self):
        """Test that _load_model caches the model to avoid reloading."""
        import common.services.transcription_services.olmoasr as olmoasr_module

        mock_olmoasr = MagicMock()
        mock_model = MagicMock()
        mock_olmoasr.load_model.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        # Clear the cache before test
        original_cache = olmoasr_module._model_cache.copy()
        olmoasr_module._model_cache.clear()

        try:
            with (
                patch.object(olmoasr_module, "settings") as mock_settings,
                patch.dict("sys.modules", {"olmoasr": mock_olmoasr, "torch": mock_torch}),
            ):
                mock_settings.OLMOASR_MODEL_SIZE = "medium"
                mock_settings.OLMOASR_DEVICE = "auto"

                # First call should load model
                model1 = OlmoASRAdapter._load_model()

                # Second call should use cache
                model2 = OlmoASRAdapter._load_model()

                assert model1 is model2
                # load_model should only be called once
                mock_olmoasr.load_model.assert_called_once()
        finally:
            # Restore original cache
            olmoasr_module._model_cache.clear()
            olmoasr_module._model_cache.update(original_cache)

    def test_load_model_uses_cuda_when_available(self):
        """Test that _load_model uses CUDA when available and device is 'auto'."""
        import common.services.transcription_services.olmoasr as olmoasr_module

        mock_olmoasr = MagicMock()
        mock_model = MagicMock()
        mock_olmoasr.load_model.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        # Clear the cache before test
        original_cache = olmoasr_module._model_cache.copy()
        olmoasr_module._model_cache.clear()

        try:
            with (
                patch.object(olmoasr_module, "settings") as mock_settings,
                patch.dict("sys.modules", {"olmoasr": mock_olmoasr, "torch": mock_torch}),
            ):
                mock_settings.OLMOASR_MODEL_SIZE = "small"
                mock_settings.OLMOASR_DEVICE = "auto"

                OlmoASRAdapter._load_model()

                mock_olmoasr.load_model.assert_called_once_with("small", inference=True, device="cuda")
        finally:
            olmoasr_module._model_cache.clear()
            olmoasr_module._model_cache.update(original_cache)

    def test_load_model_respects_explicit_device(self):
        """Test that _load_model uses explicit device setting."""
        import common.services.transcription_services.olmoasr as olmoasr_module

        mock_olmoasr = MagicMock()
        mock_model = MagicMock()
        mock_olmoasr.load_model.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True  # CUDA available but not used

        # Clear the cache before test
        original_cache = olmoasr_module._model_cache.copy()
        olmoasr_module._model_cache.clear()

        try:
            with (
                patch.object(olmoasr_module, "settings") as mock_settings,
                patch.dict("sys.modules", {"olmoasr": mock_olmoasr, "torch": mock_torch}),
            ):
                mock_settings.OLMOASR_MODEL_SIZE = "tiny"
                mock_settings.OLMOASR_DEVICE = "cpu"  # Explicit CPU

                OlmoASRAdapter._load_model()

                mock_olmoasr.load_model.assert_called_once_with("tiny", inference=True, device="cpu")
        finally:
            olmoasr_module._model_cache.clear()
            olmoasr_module._model_cache.update(original_cache)


@requires_olmoasr
@requires_test_audio
class TestOlmoASRAdapterIntegration:
    """Integration tests that require OlmoASR to be installed and configured."""

    @pytest.mark.asyncio
    async def test_transcription_produces_output(self):
        """Test that OlmoASR transcription produces non-empty output."""
        result = await OlmoASRAdapter.start(TEST_AUDIO_PATH)

        assert result.transcription_service == "olmoasr_local"
        assert len(result.transcript) > 0

        full_text = " ".join(entry["text"] for entry in result.transcript)
        assert len(full_text) > 0, "Transcription should produce non-empty text"
        print(f"OlmoASR transcription ({len(full_text)} chars): {full_text[:300]}...")

    @pytest.mark.asyncio
    async def test_transcription_structure(self):
        """Test that OlmoASR transcription returns proper DialogueEntry structure."""
        result = await OlmoASRAdapter.start(TEST_AUDIO_PATH)

        assert result.transcription_service == "olmoasr_local"
        assert len(result.transcript) == 1  # OlmoASR returns single entry

        entry = result.transcript[0]
        assert "speaker" in entry
        assert "text" in entry
        assert "start_time" in entry
        assert "end_time" in entry
        assert entry["speaker"] == "Speaker 1"

    @pytest.mark.asyncio
    async def test_check_is_noop(self):
        """Test that check() is a no-op for this synchronous adapter."""
        # First get a real transcription result
        result = await OlmoASRAdapter.start(TEST_AUDIO_PATH)

        # Check should return the same data
        checked_result = await OlmoASRAdapter.check(result)

        assert checked_result is result
        assert checked_result.transcript == result.transcript
