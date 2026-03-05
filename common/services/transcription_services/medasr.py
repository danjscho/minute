import asyncio
import dataclasses
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

# Module-level cache for loaded model and decoder
_model_cache: dict[str, tuple[Any, Any, str]] = {}
_decoder_cache: dict[str, Any] = {}


def _restore_text(text: str) -> str:
    """Restore text from pyctcdecode format back to normal format.

    After vocab preprocessing:
    - Spaces between tokens are just joiners (remove them)
    - '#' was originally '▁' (word boundary) -> convert to space
    - '</s>' is end token -> remove it
    """
    return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()


class LasrCtcBeamSearchDecoder:
    """CTC beam search decoder with language model support for MedASR/LASR models.

    This decoder wraps pyctcdecode to provide language model-guided decoding,
    which significantly improves transcription quality by using n-gram statistics.
    """

    def __init__(self, tokenizer: Any, kenlm_model_path: str | None = None, **kwargs: Any) -> None:
        import pyctcdecode

        # Use tokenizer.vocab_size which matches the model output dimension
        # (this is different from len(tokenizer.vocab) which includes extra tokens)
        vocab_size = tokenizer.vocab_size
        vocab_dict = tokenizer.vocab

        # Create vocab list sorted by index, using only up to vocab_size
        vocab = [None] * vocab_size
        for token, idx in vocab_dict.items():
            if idx < vocab_size:
                vocab[idx] = token

        # Verify no None entries
        assert not [i for i in vocab if i is None], "Vocab has missing entries"

        # pyctcdecode expects the blank label (index 0) to map to empty string
        vocab[0] = ""

        # Replace '▁' with '#' and prefix each token with a '▁'
        # This way, pyctcdecode treats each token as a "word"
        for i in range(1, len(vocab)):
            piece = vocab[i]
            if piece and not piece.startswith("<") and not piece.endswith(">"):
                piece = "▁" + piece.replace("▁", "#")
                vocab[i] = piece

        self._vocab_size = vocab_size
        self._decoder = pyctcdecode.build_ctcdecoder(vocab, kenlm_model_path, **kwargs)

    def decode_beams(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Decode with beam search and return top beams."""
        beams = self._decoder.decode_beams(*args, **kwargs)
        return [dataclasses.replace(i, text=_restore_text(i.text)) for i in beams]

    def decode(self, logits: Any, beam_width: int = 8) -> str:
        """Decode logits to text using beam search with LM."""
        beams = self.decode_beams(logits, beam_width=beam_width)
        if beams:
            return beams[0].text
        return ""


class MedASRAdapter(TranscriptionAdapter):
    """Local MedASR transcription adapter using HuggingFace transformers.

    MedASR is a CTC-based automatic speech recognition model optimized for
    medical audio. It requires:
    - transformers >= 5.0.0
    - Audio resampled to 16kHz mono
    - librosa for audio loading

    Note: MedASR provides transcription only, without speaker diarization
    or word-level timestamps.
    """

    max_audio_length = 7200  # 2 hours
    name = "medasr_local"
    adapter_type = AdapterType.SYNCHRONOUS

    @classmethod
    def is_available(cls) -> bool:
        """Check if MedASR dependencies are available."""
        if not settings.MEDASR_MODEL_NAME:
            return False
        try:
            from transformers import AutoModelForCTC  # noqa: F401

            return True
        except ImportError:
            logger.warning("transformers not available for MedASR")
            return False

    @classmethod
    def _load_model(cls) -> tuple[Any, Any, str]:
        """Lazy-load model and processor."""
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        from common.llm.adapters.huggingface import setup_huggingface_auth

        model_id = settings.MEDASR_MODEL_NAME
        if model_id in _model_cache:
            logger.info("Using cached MedASR model: %s", model_id)
            print(f"[MedASR] Using cached model: {model_id}")  # noqa: T201
            return _model_cache[model_id]

        logger.info("Loading MedASR model: %s", model_id)
        print(f"[MedASR] Loading model: {model_id}")  # noqa: T201

        # Ensure HuggingFace auth is set up before downloading gated models
        setup_huggingface_auth()

        device = settings.MEDASR_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[MedASR] Target device: {device}, CUDA available: {torch.cuda.is_available()}")  # noqa: T201

        try:
            print(f"[MedASR] Downloading processor for {model_id}...")  # noqa: T201
            processor = AutoProcessor.from_pretrained(model_id)
            print("[MedASR] Processor loaded successfully")  # noqa: T201
        except Exception as e:
            logger.exception("Failed to load MedASR processor: %s", e)
            print(f"[MedASR] FAILED to load processor: {e}")  # noqa: T201
            raise

        try:
            print(f"[MedASR] Downloading model {model_id}...")  # noqa: T201
            model = AutoModelForCTC.from_pretrained(model_id).to(device)
            print(f"[MedASR] Model loaded successfully on device: {device}")  # noqa: T201
        except Exception as e:
            logger.exception("Failed to load MedASR model: %s", e)
            print(f"[MedASR] FAILED to load model: {e}")  # noqa: T201
            raise

        _model_cache[model_id] = (model, processor, device)
        logger.info("MedASR model loaded successfully on device: %s", device)
        return model, processor, device

    @classmethod
    def _load_decoder(cls, model_id: str, tokenizer: Any) -> LasrCtcBeamSearchDecoder | None:
        """Load the CTC beam search decoder with language model.

        Returns None if pyctcdecode is not available or LM download fails.
        """
        if model_id in _decoder_cache:
            return _decoder_cache[model_id]

        if not settings.MEDASR_USE_LM:
            logger.info("Language model decoding disabled via settings")
            _decoder_cache[model_id] = None
            return None

        try:
            from huggingface_hub import hf_hub_download

            print("[MedASR] Downloading language model (lm_6.kenlm)...")  # noqa: T201
            lm_path = hf_hub_download(model_id, filename="lm_6.kenlm")
            print(f"[MedASR] Language model downloaded: {lm_path}")  # noqa: T201

            decoder = LasrCtcBeamSearchDecoder(tokenizer, lm_path)
            print("[MedASR] Beam search decoder with LM initialized")  # noqa: T201

            _decoder_cache[model_id] = decoder
            return decoder
        except ImportError:
            logger.warning("pyctcdecode not available, falling back to greedy decoding")
            print("[MedASR] pyctcdecode not available, using greedy decoding")  # noqa: T201
            _decoder_cache[model_id] = None
            return None
        except Exception as e:
            logger.warning("Failed to load language model: %s, falling back to greedy decoding", e)
            print(f"[MedASR] Failed to load LM: {e}, using greedy decoding")  # noqa: T201
            _decoder_cache[model_id] = None
            return None

    @classmethod
    def _transcribe_sync(cls, audio_path: Path) -> str:
        """Synchronous transcription (runs in thread pool)."""
        import librosa
        import torch

        model, processor, device = cls._load_model()

        # Try to load beam search decoder with language model
        model_id = settings.MEDASR_MODEL_NAME
        decoder = cls._load_decoder(model_id, processor.tokenizer)
        use_beam_search = decoder is not None

        if use_beam_search:
            print("[MedASR] Using beam search decoding with language model")  # noqa: T201
        else:
            print("[MedASR] Using greedy decoding (no language model)")  # noqa: T201

        # Load and resample audio to 16kHz mono
        logger.info("Loading audio file: %s", audio_path)
        speech, _ = librosa.load(str(audio_path), sr=16000, mono=True)

        # Process in chunks for long audio
        chunk_length = settings.MEDASR_CHUNK_LENGTH_S * 16000  # samples
        stride_length = settings.MEDASR_STRIDE_LENGTH_S * 16000

        transcripts = []
        total_samples = len(speech)
        logger.info("Processing %d samples (%.1f seconds)", total_samples, total_samples / 16000)

        for start in range(0, total_samples, chunk_length - stride_length):
            end = min(start + chunk_length, total_samples)
            chunk = speech[start:end]

            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

            with torch.inference_mode():
                if use_beam_search:
                    # Get logits for beam search decoding
                    outputs = model(**inputs)
                    logits = outputs.logits.cpu().numpy()[0]
                    # Decode with beam search and language model
                    decoded = decoder.decode(logits, beam_width=settings.MEDASR_BEAM_WIDTH)
                else:
                    # Use generate() for greedy decoding
                    outputs = model.generate(**inputs)
                    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            transcripts.append(decoded)

        return " ".join(transcripts)

    @classmethod
    async def start(cls, audio_file_path_or_recording: Path | Recording) -> TranscriptionJobMessageData:
        """Transcribe audio using MedASR.

        Args:
            audio_file_path_or_recording: Path to the audio file.
                Note: Recording objects are not supported for sync adapters.

        Returns:
            TranscriptionJobMessageData with the transcript.
        """
        if isinstance(audio_file_path_or_recording, Recording):
            msg = "MedASR sync adapter expects Path, not Recording"
            raise ValueError(msg)

        file_path = audio_file_path_or_recording
        logger.info("Starting MedASR transcription for: %s", file_path)

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        transcript_text = await loop.run_in_executor(None, partial(cls._transcribe_sync, file_path))

        # Convert to DialogueEntry format
        # MedASR doesn't provide speaker diarization or timestamps,
        # so we create a single entry with the full transcript
        dialogue_entries: list[DialogueEntry] = [
            {
                "speaker": "Speaker 1",
                "text": transcript_text,
                "start_time": 0.0,
                "end_time": 0.0,  # Unknown without additional processing
            }
        ]

        logger.info("MedASR transcription completed, %d characters", len(transcript_text))
        return TranscriptionJobMessageData(
            transcription_service=cls.name,
            transcript=dialogue_entries,
        )

    @classmethod
    async def check(cls, data: TranscriptionJobMessageData) -> TranscriptionJobMessageData:
        """No-op for synchronous adapter."""
        return data
