import os
from pathlib import Path

# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN", "your_huggingface_token_here")

# Model paths
WHISPER_MODEL_PATH = "./models/openai/whisper-small-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Audio processing settings
CHUNK_LENGTH_S = 30
STRIDE_LENGTH_S = 5

# Device settings
DEVICE = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"

# Output settings
OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)
