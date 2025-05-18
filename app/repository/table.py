from app.logger import NullLogger ## 응집성이 강화됨 주의

from logging import getLogger

from typing import Type, Optional, List, Union, get_origin, get_args
from pydantic import BaseModel
from datetime import datetime

from .client import SyncSQLiteClient, AsyncSQLiteClient
from .schema import TABLE_MODELS

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

        # 타입 추론 (Optional[int] → int)
        annotation = field.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Union and type(None) in args:
            # Optional[...] 처리
            actual_type = next((arg for arg in args if arg is not type(None)), str)
        else:
            actual_type = annotation

        db_type = PYDANTIC_TYPE_MAP.get(actual_type, "TEXT")

        col_def = f"{name} {db_type}"
        if field.default is None and field.default_factory is None:
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
    @classmethod
    def create(cls, client: SyncSQLiteClient, logging: bool = False):
        self = cls(client, logging)
        self.create_tables()
        return self

    def __init__(self, client: SyncSQLiteClient, logging: bool = False):
        self.client = client
        self.logging = logging

        if self.logging:
            self.logger = getLogger(__name__ + ".SyncTableManager")
            self.logger.info(f"SyncTableManager initialized with client: {client}")
            self.logger.info(f"TABLE_DEFINITIONS: {TABLE_DEFINITIONS}")
        else:
            self.logger = NullLogger()
        
        self.create_tables()

    def is_table_exists(self, table: str) -> bool:
        return self.client.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone() is not None

    def get_tables(self) -> List[str]:
        self.logger.debug("Get tables from client")
        return list(TABLE_DEFINITIONS.keys())

    def create_tables(self):
        self.logger.debug("Creating tables...")
        for table, sql in TABLE_DEFINITIONS.items():
            if not self.is_table_exists(table):
                self.client.execute(sql)
        self.logger.debug("Tables created successfully")

    def drop_tables(self):
        self.logger.debug("Dropping tables...")
        for table in TABLE_DEFINITIONS:
            self.client.execute(f"DROP TABLE IF EXISTS {table};")
        self.logger.debug("Tables dropped successfully")

    def insert(self, table: str, model: BaseModel) -> int:
        self.logger.debug(f"Inserting data into {table}...")
        
        data = model.model_dump(exclude_unset=True)
        keys = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = tuple(data.values())
        query = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
        cursor = self.client.execute(query, values)
        
        self.logger.debug(f"Data inserted into {table} successfully")

        return cursor.lastrowid

    def get_all(self, table: str) -> List[dict]:
        self.logger.debug(f"Getting all data from {table}...")
        rows = self.client.fetchall(f"SELECT * FROM {table}")
        self.logger.debug(f"Data fetched from {table} successfully")
        return [dict(row) for row in rows]

    def get_by_id(self, table: str, row_id: int) -> Optional[dict]:
        self.logger.debug(f"Getting data from {table} with id {row_id}...")
        rows = self.client.fetchall(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
        self.logger.debug(f"Data fetched from {table} with id {row_id} successfully")
        return dict(rows[0]) if rows else None

    def update_by_id(self, table: str, row_id: int, new_data: dict) -> bool:
        self.logger.debug(f"Updating data in {table} with id {row_id}...")
        sets = ", ".join([f"{k}=?" for k in new_data.keys()])
        values = tuple(new_data.values()) + (row_id,)
        query = f"UPDATE {table} SET {sets} WHERE id = ?"
        self.client.execute(query, values)
        self.logger.debug(f"Data updated in {table} with id {row_id} successfully")
        return True

    def delete_by_id(self, table: str, row_id: int) -> bool:
        self.logger.debug(f"Deleting data from {table} with id {row_id}...")
        self.client.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
        self.logger.debug(f"Data deleted from {table} with id {row_id} successfully")
        return True

# ─────────────────────────────
# 비동기 TableManager
# ─────────────────────────────

class AsyncTableManager:
    @classmethod
    async def create(cls, client: AsyncSQLiteClient, logging: bool = True):
        self = cls(client, logging)
        await self.create_tables()
        return self

    def __init__(self, client: AsyncSQLiteClient, logging: bool = True):
        self.client = client
        self.logging = logging

        if self.logging:
            self.logger = getLogger(__name__ + ".AsyncTableManager")
            self.logger.info(f"AsyncTableManager initialized with client: {client}")
            self.logger.info(f"TABLE_DEFINITIONS: {TABLE_DEFINITIONS}")
        else:
            self.logger = NullLogger()

    async def is_table_exists(self, table: str) -> bool:
        cursor = await self.client.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
        )
        row = await cursor.fetchone()
        return row is not None
    
    async def get_tables(self) -> List[str]:
        self.logger.debug("Get tables from client")
        return list(TABLE_DEFINITIONS.keys())
        
    async def create_tables(self):
        self.logger.debug("Creating tables...")
        for table, sql in TABLE_DEFINITIONS.items():
            if not await self.is_table_exists(table):
                await self.client.execute(sql)
        self.logger.debug("Tables created successfully")

    async def drop_tables(self):
        self.logger.debug("Dropping tables...")
        for table in TABLE_DEFINITIONS:
            await self.client.execute(f"DROP TABLE IF EXISTS {table};")
        self.logger.debug("Tables dropped successfully")

    async def insert(self, table: str, model: BaseModel) -> int:
        self.logger.debug(f"Inserting data into {table}...")

        data = model.model_dump(exclude_unset=True)
        keys = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = tuple(data.values())
        query = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
        cursor = await self.client.execute(query, values)

        self.logger.debug(f"Data inserted into {table} successfully")
        return cursor.lastrowid

    async def get_all(self, table: str) -> List[dict]:
        self.logger.debug(f"Getting all data from {table}...")
        rows = await self.client.fetchall(f"SELECT * FROM {table}")
        self.logger.debug(f"Data fetched from {table} successfully")
        return [dict(row) for row in rows]

    async def get_by_id(self, table: str, row_id: int) -> Optional[dict]:
        self.logger.debug(f"Getting data from {table} with id {row_id}...")
        rows = await self.client.fetchall(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
        self.logger.debug(f"Data fetched from {table} with id {row_id} successfully")
        return dict(rows[0]) if rows else None

    async def update_by_id(self, table: str, row_id: int, new_data: dict) -> bool:
        self.logger.debug(f"Updating data in {table} with id {row_id}...")
        sets = ", ".join([f"{k}=?" for k in new_data.keys()])
        values = tuple(new_data.values()) + (row_id,)
        query = f"UPDATE {table} SET {sets} WHERE id = ?"
        await self.client.execute(query, values)
        self.logger.debug(f"Data updated in {table} with id {row_id} successfully")
        return True

    async def delete_by_id(self, table: str, row_id: int) -> bool:
        self.logger.debug(f"Deleting data from {table} with id {row_id}...")
        await self.client.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
        self.logger.debug(f"Data deleted from {table} with id {row_id} successfully")
        return True
