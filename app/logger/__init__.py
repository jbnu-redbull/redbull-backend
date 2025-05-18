import logging

class NullLogger:
    def debug(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def critical(self, *args, **kwargs): pass
    def exception(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def isEnabledFor(self, level): return False

from .handler import start_logging, stop_logging, queue_handler

# Delay logging initialization
import atexit
def _initialize_logging():
    start_logging()
    atexit.register(stop_logging)

# Use a simple flag to ensure we only initialize once
_initialized = False
def ensure_logging_initialized():
    global _initialized
    if not _initialized:
        _initialize_logging()
        _initialized = True