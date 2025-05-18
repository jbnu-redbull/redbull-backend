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

start_logging()

import atexit
atexit.register(stop_logging)