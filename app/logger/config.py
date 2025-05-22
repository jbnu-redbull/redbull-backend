import logging

top_logger_name = __name__.split('.')[0]
top_logger = logging.getLogger(top_logger_name)

def remove_all_handlers():
    for handler in top_logger.handlers:
        top_logger.removeHandler(handler)

def disable_logging(logger: logging.Logger = None):
    """Disable logging by removing all handlers and adding NullHandler."""
    if logger is None:
        logger = top_logger
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

def create_handler(handler_type: str, level: str = "NOTSET", formatter_str: str = None, formatter_datefmt: str = None):
    if handler_type == "stream":
        handler = logging.StreamHandler()
    elif handler_type == "sqlite":
        from .handler import get_sqlite_handler
        handler = get_sqlite_handler()
    elif handler_type == "queue":
        from .queue import queue_handler
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
    
    # Always create a new formatter if formatter_str is provided
    if formatter_str:
        formatter = logging.Formatter(formatter_str, datefmt=formatter_datefmt)
        handler.setFormatter(formatter)
    # If no formatter_str but datefmt is provided, update existing formatter
    elif formatter_datefmt and handler.formatter:
        formatter = handler.formatter
        formatter.datefmt = formatter_datefmt
        handler.setFormatter(formatter)

    return handler

def add_handler(handler: logging.Handler):
    top_logger.addHandler(handler)

def set_level(level: str):
    top_logger.setLevel(level)
