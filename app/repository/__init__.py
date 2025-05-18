# app/repository/__init__.py

from .table import SyncTableManager, AsyncTableManager
from .client import SyncSQLiteClient, AsyncSQLiteClient
from .schema import TABLE_MODELS

__all__ = [
    "SyncTableManager",
    "AsyncTableManager",
    "SyncSQLiteClient",
    "AsyncSQLiteClient",
    "TABLE_MODELS"
]

sync_table_manager = SyncTableManager.create(client=SyncSQLiteClient(), logging=True)
async_table_manager = None # type: Optional[AsyncTableManager]
