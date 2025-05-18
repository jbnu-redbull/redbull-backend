# app/repository/__init__.py

from .table import SyncTableManager, AsyncTableManager
from .client import SyncSQLiteClient, AsyncSQLiteClient
from .schema import TABLE_MODELS

# Lazy initialization of managers
_sync_table_manager = None
_async_table_manager = None

def get_sync_table_manager():
    global _sync_table_manager
    if _sync_table_manager is None:
        _sync_table_manager = SyncTableManager.create(client=SyncSQLiteClient(), logging=False)
    return _sync_table_manager

def get_async_table_manager():
    global _async_table_manager
    if _async_table_manager is None:
        _async_table_manager = AsyncTableManager.create(client=AsyncSQLiteClient(), logging=True)
    return _async_table_manager

# Export the getter functions instead of the instances
__all__ = [
    'SyncTableManager',
    'AsyncTableManager',
    'SyncSQLiteClient',
    'AsyncSQLiteClient',
    'TABLE_MODELS',
    'get_sync_table_manager',
    'get_async_table_manager',
]
