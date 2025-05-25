import pytest
import os
import shutil
from pathlib import Path
from app.stt.model import initialize_diarization, initialize_asr, process_audio, align_segments
from app.stt.utils import save_results, validate_audio_file
from app.repository.client import AsyncSQLiteClient
from app.repository.table import AsyncTableManager

# Test audio file path (absolute path for testing)
TEST_AUDIO_PATH = str(Path(__file__).parent / "test_data" / "redbull.wav")
TEST_OUTPUT_PATH = str(Path(__file__).parent / "test_data" / "redbull.txt")

@pytest.fixture
def setup_test_audio():
    """Create a test audio file if it doesn't exist."""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    # Create a copy of the test audio file
    original_audio_path = Path(__file__).parent / "test_data" / "original_test_audio.wav"
    if not os.path.exists(original_audio_path):
        # Create a dummy audio file for testing
        with open(original_audio_path, "wb") as f:
            f.write(b"dummy audio content")
    
    yield
    
    # Cleanup
    # if os.path.exists(TEST_OUTPUT_PATH):
    #     os.remove(TEST_OUTPUT_PATH)

@pytest.mark.asyncio
async def test_validate_audio_file(setup_test_audio):
    """Test audio file validation."""
    assert await validate_audio_file(TEST_AUDIO_PATH) is True
    
    with pytest.raises(FileNotFoundError):
        await validate_audio_file("nonexistent.wav")
    
    with pytest.raises(ValueError):
        await validate_audio_file("tests/test_data/test.txt")

@pytest.mark.asyncio
async def test_stt_pipeline(setup_test_audio):
    """Test the complete STT pipeline."""
    # Initialize ASR pipeline only
    asr_pipeline = await initialize_asr()
    
    # Process audio with ASR only
    chunks = await process_audio(
        TEST_AUDIO_PATH,
        None,  # diarization_pipeline is None
        asr_pipeline,
        num_speakers=2
    )
    print("chunks:", chunks)  # 디버깅 로그 추가
    
    # Save results
    await save_results(chunks, TEST_OUTPUT_PATH)
    
    # Verify file was created
    assert os.path.exists(TEST_OUTPUT_PATH)
    
    # Verify database entry
    client = AsyncSQLiteClient()
    try:
        await client.connect()
        manager = AsyncTableManager(client)
        
        # Get the latest entry
        results = await manager.get_all("stt_result")
        assert len(results) > 0
        
        latest = results[-1]
        assert latest.audio_file_path == TEST_OUTPUT_PATH
        assert isinstance(latest.stt_text, str)
    finally:
        await client.close() 