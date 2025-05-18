import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from .settings import stt_settings
from app.repository.client import AsyncSQLiteClient
from app.repository.table import AsyncTableManager
from app.repository.schema import STTResult
from .table import STTTableManager

async def save_results(results: Tuple[Optional[Any], Dict[str, Any]], output_file: str) -> None:
    """Save transcription results to both a file and SQLite database."""
    # Save to file
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in results[1]['chunks']:
                f.write(f"[{entry['timestamp'][0]:.1f}-{entry['timestamp'][1] if entry['timestamp'][1] else 'end'}] {entry['text']}\n")
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to file: {e}")
        raise

    # Save to SQLite
    client = AsyncSQLiteClient(db_path=stt_settings.db_path)
    try:
        await client.connect()
        manager = STTTableManager(client)
        
        # Create tables
        await manager.create_tables()
        
        # Combine all text from results
        combined_text = " ".join([entry['text'] for entry in results[1]['chunks']])
        
        # Create STTResult instance
        stt_result = STTResult(
            audio_file_path=str(output_path),
            stt_text=combined_text
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
