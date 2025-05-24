import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from .settings import stt_settings
from app.repository.client import AsyncSQLiteClient
from app.repository.table import AsyncTableManager
from app.repository.schema import STTResult

async def save_results(results: Tuple[Optional[Any], Dict[str, Any]], output_file: str) -> None:
    """Save transcription results to both a file and SQLite database."""
    # Save to file
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def split_long_chunk(entry, max_duration=60.0):
        """Split long chunks into smaller chunks of max_duration seconds."""
        ts = entry.get('timestamp', None)
        if not (ts and isinstance(ts, (list, tuple)) and len(ts) == 2):
            return [entry]
            
        start, end = ts
        if start is None or end is None:
            return [entry]
            
        duration = end - start
        if duration <= max_duration:
            return [entry]
            
        # Split into smaller chunks
        chunks = []
        current_start = start
        while current_start < end:
            current_end = min(current_start + max_duration, end)
            # Calculate text proportion for this chunk
            chunk_ratio = (current_end - current_start) / duration
            text_length = int(len(entry['text']) * chunk_ratio)
            
            # Split text into words and take appropriate portion
            words = entry['text'].split()
            chunk_text = ' '.join(words[:text_length])
            
            chunks.append({
                'timestamp': (current_start, current_end),
                'text': chunk_text
            })
            
            current_start = current_end
            words = words[text_length:]
            
        return chunks

    def format_line(entry):
        ts = entry.get('timestamp', None)
        if ts and isinstance(ts, (list, tuple)) and len(ts) == 2:
            start = ts[0] if ts[0] is not None else 0
            end = ts[1] if ts[1] is not None else 0
            return f"[{start:.1f}-{end:.2f}]  {entry['text']}"
        return entry['text']

    # Process all chunks and split long ones
    all_chunks = []
    for entry in results[1].get('chunks', []):
        all_chunks.extend(split_long_chunk(entry))

    lines = [format_line(entry) for entry in all_chunks]
    formatted_text = "\n".join(lines)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to file: {e}")
        raise

    # Save to SQLite
    client = AsyncSQLiteClient(db_path=stt_settings.db_path)
    try:
        await client.connect()
        manager = AsyncTableManager(client)
        # Create tables
        await manager.create_tables()
        # Create STTResult instance (타임스탬프 포함 전체 텍스트 저장)
        stt_result = STTResult(
            audio_file_path=str(output_path),
            stt_text=formatted_text
        )
        # Insert into database
        row_id = await manager.insert("stt_result", stt_result)
        print(f"Results saved to database with ID: {row_id}")
    except Exception as e:
        print(f"Error saving results to database: {e}")
        raise
    finally:
        await client.close()

async def validate_audio_file(file_path: str) -> bool:
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
