import torch
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from .settings import stt_settings
from typing import Dict, List, Any, Tuple, Optional
import asyncio
from pyannote.core import Annotation

def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

async def initialize_diarization():
    """Initialize the speaker diarization pipeline."""
    device = get_device()
    print(f"Initializing diarization pipeline with device: {device}")
    
    # Run in thread pool since Pipeline.from_pretrained is blocking
    loop = asyncio.get_event_loop()
    diarization_pipeline = await loop.run_in_executor(
        None,
        lambda: Pipeline.from_pretrained(
            stt_settings.diarization_model,
            use_auth_token=stt_settings.hf_token,
            device=device
        )
    )
    return diarization_pipeline

async def initialize_asr(device=None):
    """Initialize the Whisper ASR pipeline."""
    if device is None:
        device = get_device()
        
    print(f"Initializing ASR pipeline with device: {device}")
    
    # Run in thread pool since hf_pipeline is blocking
    loop = asyncio.get_event_loop()
    asr_pipeline = await loop.run_in_executor(
        None,
        lambda: hf_pipeline(
            "automatic-speech-recognition",
            model=stt_settings.whisper_model_path,
            device=device,
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32  # Use FP16 for GPU
        )
    )
    print("ASR pipeline initialized successfully")
    return asr_pipeline

async def process_audio(
    file_path: str,
    diarization_pipeline: Optional[Pipeline] = None,
    asr_pipeline: Optional[Pipeline] = None,
    num_speakers: int = 2
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Process audio file with diarization and ASR.
    
    Args:
        file_path: Path to audio file
        diarization_pipeline: Speaker diarization pipeline
        asr_pipeline: ASR pipeline
        num_speakers: Number of speakers to detect
        
    Returns:
        Tuple of (diarization results, ASR result)
    """
    loop = asyncio.get_event_loop()
    
    # Run diarization if pipeline is provided
    diarization = None
    if diarization_pipeline is not None:
        diarization = await loop.run_in_executor(
            None,
            lambda: diarization_pipeline(file_path, num_speakers=num_speakers)
        )
    
    # Run ASR
    if asr_pipeline is None:
        raise ValueError("ASR pipeline is required")
    
    print(f"Processing audio file: {file_path}")
    result = await loop.run_in_executor(
        None,
        lambda: asr_pipeline(
            file_path,
            chunk_length_s=30,
            stride_length_s=5
        )
    )
    print("Audio processing completed")
    
    if result is None:
        raise ValueError("ASR pipeline returned None")
    
    return diarization, result

async def align_segments(diarization: Any, whisper_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Align speaker segments with transcribed text.
    
    Args:
        diarization: Diarization results
        whisper_chunks: List of transcribed chunks with timestamps
        
    Returns:
        List of aligned segments with speaker, text, and timestamps
    """
    aligned = []
    
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segment_text = []
        
        for chunk in whisper_chunks:
            if "timestamp" not in chunk:
                continue
                
            timestamp = chunk["timestamp"]
            
            if not isinstance(timestamp, tuple) or len(timestamp) != 2:
                continue
                
            chunk_start, chunk_end = timestamp
            
            if chunk_start is None or chunk_end is None:
                continue
                
            # Check for time overlap (more than 50%)
            overlap_start = max(segment.start, chunk_start)
            overlap_end = min(segment.end, chunk_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                chunk_duration = chunk_end - chunk_start
                
                if overlap_duration / chunk_duration > 0.5:
                    segment_text.append(chunk["text"])
        
        if segment_text:
            aligned.append({
                "speaker": speaker,
                "text": " ".join(segment_text),
                "start": float(segment.start),
                "end": float(segment.end)
            })
    
    return aligned
