# app/logger/handler.py

import logging
import logging.handlers
import queue
from datetime import datetime, timezone, timedelta

# --- 시간 설정 ---
KST = timezone(timedelta(hours=9))

def now_kst() -> datetime:
    return datetime.now(tz=KST)

# --- 로깅 큐 ---
log_queue = queue.Queue()

# --- SQLite용 핸들러 ---
class SQLiteHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        # Lazy import to avoid circular dependency
        from app.repository import get_sync_table_manager, TABLE_MODELS
        self.manager = get_sync_table_manager()
        self.schema = TABLE_MODELS["log"]
        self.table = "log"

    def emit(self, record):
        try:
            data = {
                "module": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "timestamp": now_kst()
            }
            model = self.schema(**data)
            self.manager.insert(self.table, model)
        except Exception:
            self.handleError(record)

# --- 리스너 설정 ---
_sqlite_handler = None
_queue_listener = None

def get_sqlite_handler():
    global _sqlite_handler
    if _sqlite_handler is None:
        _sqlite_handler = SQLiteHandler()
    return _sqlite_handler

def get_queue_listener():
    global _queue_listener
    if _queue_listener is None:
        _queue_listener = logging.handlers.QueueListener(
            log_queue,
            get_sqlite_handler(),
            respect_handler_level=True
        )
    return _queue_listener

queue_handler = logging.handlers.QueueHandler(log_queue)

# --- 리스너 컨트롤 ---
def start_logging():
    get_queue_listener().start()

def stop_logging():
    global _sqlite_handler, _queue_listener
    if _queue_listener is not None:
        _queue_listener.stop()
    if _sqlite_handler is not None:
        _sqlite_handler.close()
