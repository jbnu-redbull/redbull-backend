# app/repository/__init__.py
from .table import SyncTableManager, AsyncTableManager
from .client import SyncSQLiteClient, AsyncSQLiteClient
from .schema import TABLE_MODELS

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

def cleanup_sync_table_manager():
    """모든 리소스를 정리합니다."""
    global _sync_table_manager
    
    if _sync_table_manager is not None:
        print("SQLite Client 종료 중...")
        _sync_table_manager.client.close()
        print("SQLite Client 종료 완료")
        del _sync_table_manager
        _sync_table_manager = None

async def cleanup_async_table_manager():
    """모든 리소스를 정리합니다."""
    global _async_table_manager
    if _async_table_manager is not None:
        print("SQLite Client 종료 중...")
        await _async_table_manager.client.close()
        print("SQLite Client 종료 완료")
        del _async_table_manager
        _async_table_manager = None

async def cleanup():
    """
    FASTAPI 생명주기에 맞춰서 사용용
    """
    await cleanup_async_table_manager()
    cleanup_sync_table_manager()

# Export the getter functions instead of the instances
__all__ = [
    'SyncTableManager',
    'AsyncTableManager',
    'SyncSQLiteClient',
    'AsyncSQLiteClient',
    'TABLE_MODELS',
    'get_sync_table_manager',
    'get_async_table_manager',
    'cleanup',
]
