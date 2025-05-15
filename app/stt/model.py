import torch
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from .settings import (
    HF_TOKEN,
    WHISPER_MODEL_PATH,
    DIARIZATION_MODEL,
    CHUNK_LENGTH_S,
    STRIDE_LENGTH_S,
    DEVICE
)

def initialize_diarization():
    """Initialize the speaker diarization pipeline."""
    diarization_pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        use_auth_token=HF_TOKEN
    )
    return diarization_pipeline

def initialize_asr(device=DEVICE):
    """Initialize the Whisper ASR pipeline."""
    asr_pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_PATH,
        device=device,
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S,
        return_timestamps=True
    )
    return asr_pipeline

def process_audio(file_path, diarization_pipeline, asr_pipeline, num_speakers=None):
    """Process audio file for speaker diarization and transcription."""
    print(f"Starting speaker diarization: {file_path}")
    
    # Run speaker diarization
    if num_speakers:
        diarization = diarization_pipeline(file_path, num_speakers=num_speakers)
    else:
        diarization = diarization_pipeline(file_path)
    
    print("Starting speech recognition")
    # Run speech recognition
    transcription = asr_pipeline(
        file_path,
        generate_kwargs={"task": "transcribe", "language": "korean"}
    )
    
    return diarization, transcription["chunks"]

def align_segments(diarization, whisper_chunks):
    """Align speaker segments with transcribed text."""
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
                "start": segment.start,
                "end": segment.end
            })
    
    return aligned
