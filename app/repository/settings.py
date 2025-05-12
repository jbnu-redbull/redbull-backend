from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class SQLiteSettings(BaseSettings):
    """
    SQLite 관련 설정
    """
    db_path: str = Field("./database.db", alias="SQLITE_DB_PATH")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

sqlite_settings = SQLiteSettings()

def set_sqlite_settings(_sqlite_settings: SQLiteSettings):
    global sqlite_settings
    sqlite_settings = _sqlite_settings
