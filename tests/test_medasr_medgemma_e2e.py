"""End-to-end tests for MedASR transcription with MedGemma LLM processing.

These tests verify the full pipeline:
1. MedASR transcribes audio (no epsilon tokens)
2. MedGemma LLM generates minutes from transcription

Requires:
- GPU with CUDA support
- HF_TOKEN environment variable
- test_audio/test_audio.wav file
"""

from pathlib import Path

import pytest

from common.services.transcription_services.medasr import MedASRAdapter
from common.settings import get_settings

# Test audio file path
TEST_AUDIO_PATH = Path("test_audio/test_audio.wav")

# Mark for tests requiring local GPU models
requires_local_models = pytest.mark.skipif(
    not MedASRAdapter.is_available(),
    reason="MedASR not available (requires transformers >= 5.0.0 and MEDASR_MODEL_NAME)",
)

requires_test_audio = pytest.mark.skipif(
    not TEST_AUDIO_PATH.exists(),
    reason=f"Test audio file not found: {TEST_AUDIO_PATH}",
)


@requires_local_models
@requires_test_audio
@pytest.mark.asyncio
async def test_medasr_transcription_no_epsilon():
    """Test that MedASR transcription produces clean output without epsilon tokens."""
    result = await MedASRAdapter.start(TEST_AUDIO_PATH)

    assert result.transcription_service == "medasr_local"
    assert len(result.transcript) > 0

    # Check that transcription text doesn't contain epsilon tokens
    full_text = " ".join(entry["text"] for entry in result.transcript)
    assert "<epsilon>" not in full_text, f"Transcription contains epsilon tokens: {full_text[:200]}..."
    assert "</s>" not in full_text, f"Transcription contains end tokens: {full_text[:200]}..."

    # Verify we got meaningful text
    assert len(full_text) > 50, f"Transcription too short: {full_text}"
    print(f"MedASR transcription ({len(full_text)} chars): {full_text[:300]}...")


@requires_local_models
@requires_test_audio
@pytest.mark.asyncio
async def test_medasr_transcription_structure():
    """Test that MedASR transcription returns proper DialogueEntry structure."""
    result = await MedASRAdapter.start(TEST_AUDIO_PATH)

    assert result.transcription_service == "medasr_local"
    assert len(result.transcript) == 1  # MedASR returns single entry

    entry = result.transcript[0]
    assert "speaker" in entry
    assert "text" in entry
    assert "start_time" in entry
    assert "end_time" in entry
    assert entry["speaker"] == "Speaker 1"


@requires_local_models
@pytest.mark.asyncio
async def test_medgemma_llm_available():
    """Test that MedGemma LLM adapter is available and can be initialized."""
    settings = get_settings()

    # Check if HuggingFace provider is configured
    if settings.FAST_LLM_PROVIDER != "huggingface":
        pytest.skip("FAST_LLM_PROVIDER is not set to 'huggingface'")

    from common.llm.adapters.huggingface import HuggingFaceModelAdapter

    adapter = HuggingFaceModelAdapter(
        model=settings.FAST_LLM_MODEL_NAME,
        device=settings.LOCAL_LLM_DEVICE,
        temperature=0.0,
        max_new_tokens=256,
    )

    # Test a simple chat completion
    messages = [{"role": "user", "content": "Say 'hello' and nothing else."}]
    response = await adapter.chat(messages)

    assert response is not None
    assert len(response) > 0
    print(f"MedGemma response: {response}")


@requires_local_models
@requires_test_audio
@pytest.mark.asyncio
async def test_medasr_to_medgemma_pipeline():
    """Test the full pipeline: MedASR transcription -> MedGemma summarization."""
    settings = get_settings()

    if settings.FAST_LLM_PROVIDER != "huggingface":
        pytest.skip("FAST_LLM_PROVIDER is not set to 'huggingface'")

    # Step 1: Transcribe with MedASR
    print("Step 1: Running MedASR transcription...")
    transcription_result = await MedASRAdapter.start(TEST_AUDIO_PATH)

    full_text = " ".join(entry["text"] for entry in transcription_result.transcript)
    assert "<epsilon>" not in full_text, "Transcription contains epsilon tokens"
    print(f"Transcription complete: {len(full_text)} characters")

    # Step 2: Summarize with MedGemma
    print("Step 2: Running MedGemma summarization...")
    from common.llm.adapters.huggingface import HuggingFaceModelAdapter

    adapter = HuggingFaceModelAdapter(
        model=settings.FAST_LLM_MODEL_NAME,
        device=settings.LOCAL_LLM_DEVICE,
        temperature=0.0,
        max_new_tokens=512,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a medical transcription assistant. Summarize the following medical transcription concisely.",
        },
        {"role": "user", "content": f"Please summarize this medical transcription:\n\n{full_text}"},
    ]

    summary = await adapter.chat(messages)

    assert summary is not None
    assert len(summary) > 50, f"Summary too short: {summary}"
    print(f"Summary ({len(summary)} chars): {summary}")

    # Verify the summary doesn't contain epsilon artifacts
    assert "<epsilon>" not in summary
    assert "epsilon" not in summary.lower() or "epsilon" in full_text.lower()


@requires_local_models
@requires_test_audio
@pytest.mark.asyncio
async def test_medasr_chunked_processing():
    """Test that MedASR properly handles chunked audio processing."""
    settings = get_settings()

    # Get chunk settings
    chunk_length = settings.MEDASR_CHUNK_LENGTH_S
    stride_length = settings.MEDASR_STRIDE_LENGTH_S

    print(f"Chunk settings: chunk_length={chunk_length}s, stride_length={stride_length}s")

    result = await MedASRAdapter.start(TEST_AUDIO_PATH)

    full_text = " ".join(entry["text"] for entry in result.transcript)

    # Verify no epsilon tokens in chunked output
    assert "<epsilon>" not in full_text
    assert len(full_text) > 0

    print(f"Chunked transcription result: {full_text[:300]}...")
