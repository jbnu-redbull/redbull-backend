# app/logger/handler.py

import logging
from datetime import datetime, timezone, timedelta

# --- 시간 설정 ---
KST = timezone(timedelta(hours=9))

def now_kst() -> datetime:
    return datetime.now(tz=KST)

# --- SQLite용 핸들러 ---
class SQLiteHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self._manager = None
        self._schema = None
        self._table = "log"

    def __del__(self):
        if self._manager is not None:
            del self._manager
            self._manager = None

    @property
    def manager(self):
        if self._manager is None:
            # Lazy import to ensure database connection is created in the correct thread
            from .database import get_sync_table_manager
            self._manager = get_sync_table_manager()
        return self._manager

    @property
    def schema(self):
        if self._schema is None:
            # Lazy import to ensure schema is loaded in the correct thread
            from .database import Log
            self._schema = Log
        return self._schema

    def emit(self, record):
        try:
            data = {
                "module": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "timestamp": now_kst()
            }
            model = self.schema(**data)
            self.manager.insert(model)
        except Exception:
            self.handleError(record)

# --- SQLite 핸들러 싱글톤 ---
_sqlite_handler = None

def set_sqlite_handler(handler: SQLiteHandler):
    global _sqlite_handler
    _sqlite_handler = handler

def get_sqlite_handler():
    global _sqlite_handler
    if _sqlite_handler is None:
        _sqlite_handler = SQLiteHandler()
    return _sqlite_handler