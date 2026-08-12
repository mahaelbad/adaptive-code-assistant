"""
Logging Configuration
"""

import logging
from pathlib import Path


def setup_logger(name: str = "AdaptiveCodeAssistant") -> logging.Logger:
    """
    Configure and return a project logger.
    """

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # File Handler
    file_handler = logging.FileHandler(
        log_dir / "project.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger