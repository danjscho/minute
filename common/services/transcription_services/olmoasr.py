import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import Any

from common.database.postgres_models import DialogueEntry, Recording
from common.services.transcription_services.adapter import AdapterType, TranscriptionAdapter
from common.settings import get_settings
from common.types import TranscriptionJobMessageData

settings = get_settings()
logger = logging.getLogger(__name__)

# Module-level cache for loaded model
_model_cache: dict[str, Any] = {}


class OlmoASRAdapter(TranscriptionAdapter):
    """Local OlmoASR transcription adapter using Allen AI's olmoasr package.

    OlmoASR is a Whisper-like encoder-decoder ASR model trained on 440K hours
    of weakly-supervised audio-text pairs. It requires:
    - olmoasr package (installed from GitHub)
    - ffmpeg for audio processing

    Model sizes available: tiny (39M), base (74M), small (244M),
    medium (769M), large (1.5B), large-v2 (1.5B)

    Note: OlmoASR provides transcription only, without speaker diarization.
    """

    max_audio_length = 7200  # 2 hours
    name = "olmoasr_local"
    adapter_type = AdapterType.SYNCHRONOUS

    @classmethod
    def is_available(cls) -> bool:
        """Check if OlmoASR dependencies are available."""
        if not settings.OLMOASR_MODEL_SIZE:
            return False
        try:
            import olmoasr  # noqa: F401

            return True
        except ImportError:
            logger.warning("olmoasr package not available")
            return False

    @classmethod
    def _load_model(cls) -> Any:
        """Lazy-load OlmoASR model with caching."""
        import torch

        model_size = settings.OLMOASR_MODEL_SIZE
        if model_size in _model_cache:
            logger.info("Using cached OlmoASR model: %s", model_size)
            print(f"[OlmoASR] Using cached model: {model_size}")  # noqa: T201
            return _model_cache[model_size]

        logger.info("Loading OlmoASR model: %s", model_size)
        print(f"[OlmoASR] Loading model: {model_size}")  # noqa: T201

        device = settings.OLMOASR_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[OlmoASR] Target device: {device}, CUDA available: {torch.cuda.is_available()}")  # noqa: T201

        try:
            import olmoasr

            print(f"[OlmoASR] Downloading model {model_size}...")  # noqa: T201
            model = olmoasr.load_model(model_size, inference=True, device=device)
            print(f"[OlmoASR] Model loaded successfully on device: {device}")  # noqa: T201
        except Exception as e:
            logger.exception("Failed to load OlmoASR model: %s", e)
            print(f"[OlmoASR] FAILED to load model: {e}")  # noqa: T201
            raise

        _model_cache[model_size] = model
        logger.info("OlmoASR model loaded successfully on device: %s", device)
        return model

    @classmethod
    def _transcribe_sync(cls, audio_path: Path) -> str:
        """Synchronous transcription (runs in thread pool)."""
        model = cls._load_model()

        logger.info("Starting OlmoASR transcription for: %s", audio_path)
        print(f"[OlmoASR] Transcribing: {audio_path}")  # noqa: T201

        try:
            result = model.transcribe(str(audio_path))
            # Result may be a string or dict with 'text' key depending on version
            if isinstance(result, dict):
                transcript_text = result.get("text", "")
            else:
                transcript_text = str(result)

            print(f"[OlmoASR] Transcription completed, {len(transcript_text)} characters")  # noqa: T201
            return transcript_text
        except Exception as e:
            logger.exception("OlmoASR transcription failed: %s", e)
            print(f"[OlmoASR] Transcription FAILED: {e}")  # noqa: T201
            raise

    @classmethod
    async def start(cls, audio_file_path_or_recording: Path | Recording) -> TranscriptionJobMessageData:
        """Transcribe audio using OlmoASR.

        Args:
            audio_file_path_or_recording: Path to the audio file.
                Note: Recording objects are not supported for sync adapters.

        Returns:
            TranscriptionJobMessageData with the transcript.
        """
        if isinstance(audio_file_path_or_recording, Recording):
            msg = "OlmoASR sync adapter expects Path, not Recording"
            raise ValueError(msg)

        file_path = audio_file_path_or_recording
        logger.info("Starting OlmoASR transcription for: %s", file_path)

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        transcript_text = await loop.run_in_executor(None, partial(cls._transcribe_sync, file_path))

        # Convert to DialogueEntry format
        # OlmoASR doesn't provide speaker diarization,
        # so we create a single entry with the full transcript
        dialogue_entries: list[DialogueEntry] = [
            {
                "speaker": "Speaker 1",
                "text": transcript_text,
                "start_time": 0.0,
                "end_time": 0.0,  # Unknown without additional processing
            }
        ]

        logger.info("OlmoASR transcription completed, %d characters", len(transcript_text))
        return TranscriptionJobMessageData(
            transcription_service=cls.name,
            transcript=dialogue_entries,
        )

    @classmethod
    async def check(cls, data: TranscriptionJobMessageData) -> TranscriptionJobMessageData:
        """No-op for synchronous adapter."""
        return data
