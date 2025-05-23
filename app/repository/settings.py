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

sqlite_settings = SQLiteSettings()

def set_sqlite_settings(_sqlite_settings: SQLiteSettings):
    global sqlite_settings
    sqlite_settings = _sqlite_settings
    return sqlite_settings