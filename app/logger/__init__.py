import atexit
import time

from .settings import logger_setup

def initialize_logging():
    logger_setup()

from .queue import stop_queue_listener

def cleanup():
    print("Starting cleanup...")
    # 큐 리스너를 중지하고 완전히 종료될 때까지 대기
    stop_queue_listener()
    print("Cleanup completed.")

atexit.register(cleanup)

__all__ = ["initialize_logging"]