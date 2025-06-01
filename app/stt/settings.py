from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class STTSettings(BaseSettings):
    """
    STT(Speech-to-Text) 관련 설정
    """
    # Hugging Face Token
    hf_token: str = Field("your_huggingface_token_here", alias="HF_TOKEN")
    
    # Model paths
    whisper_model: str = Field("alicekyting/whisper-large-v3-4bit", alias="WHISPER_MODEL")
    diarization_model: str = Field("pyannote/speaker-diarization-3.1", alias="DIARIZATION_MODEL")
    
    # Audio processing settings
    chunk_length_s: int = Field(15, alias="CHUNK_LENGTH_S")
    stride_length_s: int = Field(3, alias="STRIDE_LENGTH_S")
    
    # Device settings
    device: str = Field("cpu", alias="DEVICE")
    #device: str = Field("cuda", alias="DEVICE")
    
    # Output settings
    output_dir: Path = Field(Path("./results"), alias="OUTPUT_DIR")
    
    # Database settings
    db_path: str = Field("./database.db", alias="DB_PATH")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_dir.mkdir(exist_ok=True)

# 전역 설정 객체 생성
stt_settings = STTSettings()

def set_stt_settings(_stt_settings: STTSettings):
    """
    STT 설정을 전역적으로 변경하기 위한 함수
    """
    global stt_settings
    stt_settings = _stt_settings
