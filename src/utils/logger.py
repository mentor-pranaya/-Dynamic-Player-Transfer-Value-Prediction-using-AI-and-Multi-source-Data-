
import logging, os, sys

def get_logger(name: str = "transferiq", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
