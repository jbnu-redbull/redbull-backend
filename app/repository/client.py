import sqlite3
import aiosqlite
from typing import Optional, Any
from .settings import sqlite_settings

SQLITE_DB_PATH = sqlite_settings.db_path

class SyncSQLiteClient:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or SQLITE_DB_PATH
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # 딕셔너리처럼 결과 반환

    def execute(self, query: str, params: tuple = ()) -> Any:
        if self.conn is None:
            self.connect()
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor

    def fetchall(self, query: str, params: tuple = ()) -> list:
        cursor = self.execute(query, params)
        return cursor.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


class AsyncSQLiteClient:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or SQLITE_DB_PATH
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row

    async def execute(self, query: str, params: tuple = ()) -> aiosqlite.Cursor:
        if self.conn is None:
            await self.connect()
        cursor = await self.conn.execute(query, params)
        await self.conn.commit()
        return cursor

    async def fetchall(self, query: str, params: tuple = ()) -> list:
        cursor = await self.execute(query, params)
        rows = await cursor.fetchall()
        return rows

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
