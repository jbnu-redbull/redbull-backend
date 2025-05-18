import logging
from .handler import queue_handler

top_logger_name = __name__.split('.')[0]
top_logger = logging.getLogger(top_logger_name)

def remove_all_handlers():
    for handler in top_logger.handlers:
        top_logger.removeHandler(handler)

def create_handler(handler_type: str, level: str = "NOTSET", formatter_str: str = None, formatter_datefmt: str = None):
    if handler_type == "stream":
        handler = logging.StreamHandler()
    elif handler_type == "queue":
        handler = queue_handler
    else:
        raise ValueError(f"Invalid handler type: {handler_type}")

    if level == "NOTSET":
        level = logging.NOTSET
    else:
        try:
            level = getattr(logging, level)
        except AttributeError:
            raise ValueError(f"Invalid level: {level}")

    handler.setLevel(level)
    if formatter_str:
        formatter = logging.Formatter(formatter_str)
    else:
        formatter = handler.formatter
    
    if formatter_datefmt:
        formatter.datefmt = formatter_datefmt

    handler.setFormatter(formatter)

    return handler

def add_handler(handler: logging.Handler):
    top_logger.addHandler(handler)

def set_level(level: str):
    top_logger.setLevel(level)
