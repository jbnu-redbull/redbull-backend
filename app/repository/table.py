from typing import Type, Optional, List, Any
from pydantic import BaseModel
from datetime import datetime

from .client import SyncSQLiteClient, AsyncSQLiteClient
from .schema import TABLE_MODELS
from app.repository.schema import STTResult

# ─────────────────────────────
# SQL 생성 유틸
# ─────────────────────────────

PYDANTIC_TYPE_MAP = {
    int: "INTEGER",
    str: "TEXT",
    float: "REAL",
    bool: "BOOLEAN",
    datetime: "TEXT",
}

def generate_create_sql(model: Type[BaseModel], table_name: str) -> str:
    fields = []
    for name, field in model.model_fields.items():
        if name == "id":
            fields.append("id INTEGER PRIMARY KEY AUTOINCREMENT")
            continue

        type_ = field.annotation
        origin = getattr(type_, "__origin__", type_)
        db_type = PYDANTIC_TYPE_MAP.get(origin, "TEXT")

        col_def = f"{name} {db_type}"
        if field.is_required():
            col_def += " NOT NULL"
        fields.append(col_def)

    fields_sql = ",\n    ".join(fields)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {fields_sql}\n);"

TABLE_DEFINITIONS = {
    table_name: generate_create_sql(model, table_name)
    for table_name, model in TABLE_MODELS.items()
}

# ─────────────────────────────
# 동기 TableManager
# ─────────────────────────────

class SyncTableManager:
    def __init__(self, client: SyncSQLiteClient):
        self.client = client

    def create_tables(self):
        for table, sql in TABLE_DEFINITIONS.items():
            self.client.execute(sql)

    def drop_tables(self):
        for table in TABLE_DEFINITIONS:
            self.client.execute(f"DROP TABLE IF EXISTS {table};")

    def insert(self, table: str, model: BaseModel) -> int:
        data = model.model_dump(exclude_unset=True)
        keys = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = tuple(data.values())
        query = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
        cursor = self.client.execute(query, values)
        return cursor.lastrowid

    def get_all(self, table: str) -> List[dict]:
        rows = self.client.fetchall(f"SELECT * FROM {table}")
        return [dict(row) for row in rows]

    def get_by_id(self, table: str, row_id: int) -> Optional[dict]:
        rows = self.client.fetchall(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
        return dict(rows[0]) if rows else None

    def update_by_id(self, table: str, row_id: int, new_data: dict) -> bool:
        sets = ", ".join([f"{k}=?" for k in new_data.keys()])
        values = tuple(new_data.values()) + (row_id,)
        query = f"UPDATE {table} SET {sets} WHERE id = ?"
        self.client.execute(query, values)
        return True

    def delete_by_id(self, table: str, row_id: int) -> bool:
        self.client.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
        return True

# ─────────────────────────────
# 비동기 TableManager
# ─────────────────────────────

class AsyncTableManager:
    def __init__(self, client: AsyncSQLiteClient):
        self.client = client

    async def create_tables(self):
        for table, sql in TABLE_DEFINITIONS.items():
            await self.client.execute(sql)

    async def drop_tables(self):
        for table in TABLE_DEFINITIONS:
            await self.client.execute(f"DROP TABLE IF EXISTS {table};")

    async def insert(self, table: str, model: BaseModel) -> int:
        data = model.model_dump(exclude_unset=True)
        keys = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = tuple(data.values())
        query = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
        cursor = await self.client.execute(query, values)
        return cursor.lastrowid

    async def get_all(self, table: str) -> List[Any]:
        """Get all rows from a table."""
        query = f"SELECT * FROM {table}"
        cursor = await self.client.execute(query)
        rows = await cursor.fetchall()
        
        if table == "stt_result":
            return [STTResult(**row) for row in rows]
        return rows

    async def get_by_id(self, table: str, row_id: int) -> Optional[dict]:
        rows = await self.client.fetchall(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
        return dict(rows[0]) if rows else None

    async def update_by_id(self, table: str, row_id: int, new_data: dict) -> bool:
        sets = ", ".join([f"{k}=?" for k in new_data.keys()])
        values = tuple(new_data.values()) + (row_id,)
        query = f"UPDATE {table} SET {sets} WHERE id = ?"
        await self.client.execute(query, values)
        return True

    async def delete_by_id(self, table: str, row_id: int) -> bool:
        await self.client.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
        return True
