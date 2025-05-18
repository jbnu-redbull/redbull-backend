# app/logger/handler.py

import logging
import logging.handlers
import queue
from datetime import datetime, timezone, timedelta

from app.repository import sync_table_manager, TABLE_MODELS

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
        self.manager = sync_table_manager
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
sqlite_handler = SQLiteHandler()
queue_listener = logging.handlers.QueueListener(
    log_queue,
    sqlite_handler,
    respect_handler_level=True
)

queue_handler = logging.handlers.QueueHandler(log_queue)

# --- 리스너 컨트롤 ---
def start_logging():
    queue_listener.start()

def stop_logging():
    queue_listener.stop()
    sqlite_handler.close()
