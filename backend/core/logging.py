"""Centralized logging initialization.

Provides a single ``setup_logging`` function that configures
the root application logger.  Call once at startup.
"""

import logging
import sys


def setup_logging(log_level: str = "DEBUG", log_file: str = "dss.log") -> logging.Logger:
    """Configure and return the application-wide logger.

    Parameters
    ----------
    log_level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
    log_file:
        Name of the file in ``backend/logs/`` (reserved for future file handler).

    Returns
    -------
    logging.Logger
        The configured root logger for the ``dss`` namespace.
    """
    logger = logging.getLogger("dss")
    logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False

    return logger
