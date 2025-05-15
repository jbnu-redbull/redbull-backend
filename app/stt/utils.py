import os
from pathlib import Path
from typing import List, Dict, Any
from .settings import OUTPUT_DIR

def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save transcription results to a file."""
    output_path = OUTPUT_DIR / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(f"[{entry['start']:.1f}-{entry['end']:.1f}] {entry['speaker']}: {entry['text']}\n")
    print(f"Results saved to {output_path}")

def validate_audio_file(file_path: str) -> bool:
    """Validate if the audio file exists and has a supported format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    supported_formats = ['.wav', '.mp3', '.m4a', '.flac']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(supported_formats)}")
    
    return True

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
