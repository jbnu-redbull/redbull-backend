import queue
import logging
import logging.handlers
from typing import List, Optional

# --- 로깅 큐 ---
log_queue = queue.Queue()

# --- 큐 핸들러 ---
queue_handler = logging.handlers.QueueHandler(log_queue)

# --- 큐 리스너 ---
_queue_listener: Optional[logging.handlers.QueueListener] = None

def set_queue_listener(handlers: List[logging.Handler]) -> logging.handlers.QueueListener:
    """Build a new queue listener with multiple handlers."""
    global _queue_listener
    
    if not handlers:
        raise ValueError("At least one handler must be provided")
    
    _queue_listener = logging.handlers.QueueListener(
        log_queue,
        *handlers,  # 핸들러 리스트를 언패킹
        respect_handler_level=True
    )

def get_queue_listener() -> Optional[logging.handlers.QueueListener]:
    """Get the existing queue listener."""
    return _queue_listener

def is_queue_listener_running() -> bool:
    """Check if the queue listener is currently running."""
    return _queue_listener is not None and _queue_listener._thread is not None and _queue_listener._thread.is_alive()

def start_queue_listener():
    """Start the queue listener if it's not already running."""
    if not is_queue_listener_running():
        if _queue_listener is None:
            raise RuntimeError("Queue listener not initialized. Call logger_setup() first.")
        _queue_listener.start()

def stop_queue_listener():
    """Stop the queue listener if it's running."""
    global _queue_listener
    if is_queue_listener_running():
        _queue_listener.stop()
        if _queue_listener._thread is not None:
            _queue_listener._thread.join(timeout=1.0)
        _queue_listener = None
