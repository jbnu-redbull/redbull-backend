from .config import (
    add_handler,
    create_handler,
    set_level,
    remove_all_handlers,
)

from .handler import (
    SQLiteHandler,
)

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class LoggerSettings(BaseSettings):
    """
    Logger 관련 설정
    """
    logger_level: str = Field("INFO", alias="LOGGER_LEVEL")
    logger_db_logging: bool = Field(False, alias="LOGGER_DB_LOGGING")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

logger_settings = LoggerSettings()

def logger_setup(_logger_settings: LoggerSettings = None):
    global logger_settings  

    if _logger_settings is None:
        _logger_settings = logger_settings
    
    set_level(_logger_settings.logger_level)

    remove_all_handlers()
    
    stream_handler = create_handler(
        handler_type="stream", 
        formatter_str= "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
        formatter_datefmt = "%Y-%m-%d %H:%M:%S"
    )
    add_handler(stream_handler)
    
    if _logger_settings.logger_db_logging:
        db_handler = create_handler(
            handler_type="queue",
            level=_logger_settings.logger_level,
            formatter_str= "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
            formatter_datefmt = "%Y-%m-%d %H:%M:%S"
        )
        add_handler(db_handler)

def set_logger_settings(_logger_settings: LoggerSettings):
    global logger_settings
    logger_settings = _logger_settings
    logger_setup()
