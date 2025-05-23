# app/logger/settings.py
import logging
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class LoggerSettings(BaseSettings):
    """
    Logger 관련 설정
    """
    logger_level: str = Field("INFO", alias="LOGGER_LEVEL")
    logger_db_path: str = Field("./database.db", alias="LOGGER_DB_PATH")
    logger_db_logging: bool = Field(False, alias="LOGGER_DB_LOGGING")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
        use_enum_values=True,
        populate_by_name=True
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 생성자로 전달된 값이 우선되도록 설정
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

logger_settings = LoggerSettings()

# 기본 포맷터 설정
DEFAULT_FORMATTER = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

def logger_setup():
    """로깅 시스템을 초기화합니다."""
    from .config import (
        add_handler,
        create_handler,
        set_level,
        remove_all_handlers,
    )
    from .queue import (
        start_queue_listener,
        stop_queue_listener,
        set_queue_listener
    )
    try:
        set_level(logger_settings.logger_level)
        remove_all_handlers()

        # Queue Listener에 붙일 핸들러 종류(여기에 Queue Handler를 추가해선 안됨(순환 호출 구조))
        handlers = []

        # Stream 핸들러 추가
        stream_handler = create_handler(
            handler_type="stream", 
            formatter_str=DEFAULT_FORMATTER,
            formatter_datefmt=DEFAULT_DATEFMT
        )
        handlers.append(stream_handler)
        
        # DB 핸들러 추가 (설정된 경우)
        if logger_settings.logger_db_logging:
            db_handler = create_handler(
                handler_type="sqlite",
                level=logger_settings.logger_level,
                formatter_str=DEFAULT_FORMATTER,
                formatter_datefmt=DEFAULT_DATEFMT
            )
            handlers.append(db_handler)
        
        # 기존 리스너 정리
        stop_queue_listener()
        
        # 새로운 리스너 설정
        set_queue_listener(handlers)
        
        # 큐 핸들러 생성 및 추가 (포맷터 없이)
        queue_handler = create_handler(
            handler_type="queue",
            level=logger_settings.logger_level
        )
        add_handler(queue_handler)
        
        # 리스너 시작
        start_queue_listener()
        
    except Exception as e:
        logging.error(f"Failed to setup logging: {e}")
        raise

def set_logger_settings(new_settings: LoggerSettings) -> LoggerSettings:
    """로거 설정을 업데이트하고 로깅 시스템을 재초기화합니다."""
    global logger_settings
    
    # 새로운 설정으로 업데이트
    logger_settings = new_settings
    
    # 로깅 시스템 재초기화
    logger_setup()
    
    return logger_settings
