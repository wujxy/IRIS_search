"""
IRIS Logging Configuration

Unified logging setup for the IRIS project.
Provides consistent logging format across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Colored format for console
CONSOLE_FORMAT = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"


def setup_logging(
    level: str | int = "INFO",
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    console: bool = True,
    clear_handlers: bool = False,
) -> None:
    """
    Set up logging for the IRIS application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        format_string: Custom format string
        date_format: Custom date format string
        console: Whether to add console handler
        clear_handlers: Clear existing handlers before setup
    """
    # Convert level string to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers if requested
    if clear_handlers:
        root_logger.handlers.clear()

    # Use default formats if not provided
    if format_string is None:
        format_string = DEFAULT_FORMAT
    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)

    # Add console handler
    if console and not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_level(level: str | int, logger: Optional[logging.Logger] = None) -> None:
    """
    Set logging level for a logger.

    Args:
        level: Logging level
        logger: Logger to set level for (defaults to root logger)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    target_logger = logger or logging.getLogger()
    target_logger.setLevel(level)


def disable_third_party_logging(*loggers: str) -> None:
    """
    Disable logging for specified third-party loggers.

    Args:
        *loggers: Names of loggers to disable
    """
    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)


# Common third-party loggers to silence
QUIET_LOGGERS = [
    "uvicorn.access",
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
]


def setup_quiet_logging(level: str = "WARNING", log_file: Optional[str] = None) -> None:
    """
    Set up quiet logging (less verbose).

    Args:
        level: Logging level for application logs
        log_file: Optional file path for logs
    """
    setup_logging(level=level, log_file=log_file, clear_handlers=True)
    disable_third_party_logging(*QUIET_LOGGERS)
