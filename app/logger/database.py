import sqlite3
from typing import Optional, Any, List, Dict
from datetime import datetime
import logging

from .settings import logger_settings
from pydantic import BaseModel, Field

_sync_table_manager = None

class Log(BaseModel):
    id: Optional[int] = Field(None, description="로그 ID")
    module: str = Field(..., description="모듈 이름")
    level: str = Field(..., description="로그 레벨")
    message: str = Field(..., description="로그 메시지")
    timestamp: datetime = Field(..., description="로그 시간")

class SyncSQLiteClient:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or logger_settings.logger_db_path
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
                self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to connect to database: {e}")

    def execute(self, query: str, params: tuple = ()) -> Any:
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            return cursor
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to execute query: {e}")

    def fetchall(self, query: str, params: tuple = ()) -> List[Dict]:
        try:
            cursor = self.execute(query, params)
            return cursor.fetchall()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to fetch data: {e}")

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except sqlite3.Error as e:
                raise RuntimeError(f"Failed to close connection: {e}")
            finally:
                self.conn = None

class SyncTableManager:
    def __init__(self):
        self.model = Log

    def _get_client(self) -> SyncSQLiteClient:
        return SyncSQLiteClient()

    def is_table_exists(self, table: str) -> bool:
        try:
            with self._get_client() as client:
                return client.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                ).fetchone() is not None
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to check table existence: {e}")

    def create_log_table(self):
        if not self.is_table_exists("log"):
            try:
                with self._get_client() as client:
                    client.execute("""
                        CREATE TABLE IF NOT EXISTS log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            module TEXT NOT NULL,
                            level TEXT NOT NULL,
                            message TEXT NOT NULL,
                            timestamp DATETIME NOT NULL
                        )
                    """)
            except sqlite3.Error as e:
                raise RuntimeError(f"Failed to create log table: {e}")

    def drop_log_table(self):
        if self.is_table_exists("log"):
            try:
                with self._get_client() as client:
                    client.execute("DROP TABLE IF EXISTS log;")
            except sqlite3.Error as e:
                raise RuntimeError(f"Failed to drop log table: {e}")

    def insert(self, model: Log) -> int:
        try:
            data = model.model_dump(exclude_unset=True)
            keys = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            values = tuple(data.values())
            query = f"INSERT INTO log ({keys}) VALUES ({placeholders})"
            with self._get_client() as client:
                cursor = client.execute(query, values)
                return cursor.lastrowid
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to insert log: {e}")

    def get_all(self) -> List[Dict]:
        try:
            with self._get_client() as client:
                rows = client.fetchall("SELECT * FROM log")
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to get all logs: {e}")

    def get_by_id(self, row_id: int) -> Optional[Dict]:
        try:
            with self._get_client() as client:
                rows = client.fetchall("SELECT * FROM log WHERE id = ?", (row_id,))
                return dict(rows[0]) if rows else None
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to get log by id: {e}")

    def update_by_id(self, row_id: int, new_data: Dict) -> bool:
        try:
            sets = ", ".join([f"{k}=?" for k in new_data.keys()])
            values = tuple(new_data.values()) + (row_id,)
            query = f"UPDATE log SET {sets} WHERE id = ?"
            with self._get_client() as client:
                client.execute(query, values)
                return True
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to update log: {e}")

    def delete_by_id(self, row_id: int) -> bool:
        try:
            with self._get_client() as client:
                client.execute("DELETE FROM log WHERE id = ?", (row_id,))
                return True
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to delete log: {e}")

def get_sync_table_manager():
    global _sync_table_manager
    if _sync_table_manager is None:
        _sync_table_manager = SyncTableManager()
    return _sync_table_manager

def close_sync_table_manager():
    """Close the sync table manager connection."""
    global _sync_table_manager
    _sync_table_manager = None