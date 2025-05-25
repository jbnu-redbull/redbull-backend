import torch
from pyannote.audio import Pipeline
import subprocess
import json
import os
from pathlib import Path
from .settings import stt_settings
from typing import Dict, List, Any, Tuple, Optional
import asyncio
from pyannote.core import Annotation

def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        print("CUDA is available")
        return "cuda"
    else:
        print("CUDA is not available")
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

async def process_audio(
    file_path: str,
    diarization_pipeline: Optional[Pipeline] = None,
    num_speakers: int = 2
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Process audio file with diarization and ASR using whisper.cpp.
    
    Args:
        file_path: Path to audio file
        diarization_pipeline: Speaker diarization pipeline
        num_speakers: Number of speakers to detect
        
    Returns:
        Tuple of (diarization results, ASR result)
    """
    # Check if whisper.cpp and model exist
    whisper_main = stt_settings.whisper_cpp_dir / "main"
    if not whisper_main.exists():
        raise FileNotFoundError(f"whisper.cpp main executable not found at {whisper_main}")
    if not stt_settings.whisper_model_path.exists():
        raise FileNotFoundError(f"whisper model not found at {stt_settings.whisper_model_path}")

    loop = asyncio.get_event_loop()
    
    # Run diarization if pipeline is provided
    diarization = None
    if diarization_pipeline is not None:
        diarization = await loop.run_in_executor(
            None,
            lambda: diarization_pipeline(file_path, num_speakers=num_speakers)
        )
    
    # Run whisper.cpp
    print(f"Processing audio file: {file_path}")
    
    # Create output file path
    output_base = os.path.splitext(file_path)[0]
    
    # Run whisper.cpp command
    cmd = [
        str(whisper_main),
        "-m", str(stt_settings.whisper_model_path),
        "-t", "4",  # number of threads
        "-f", file_path,
        "-of", output_base,
        "-otxt",
        "-nt",  # no timestamps
        "-np"   # no progress
    ]
    
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(cmd, capture_output=True, text=True)
    )
    
    if result.returncode != 0:
        raise ValueError(f"Whisper.cpp failed: {result.stderr}")
    
    # Read the output file
    with open(f"{output_base}.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Format result similar to the original whisper output
    result = {
        "text": text,
        "chunks": [{"text": text, "timestamp": None}]
    }
    
    print("Audio processing completed")
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
