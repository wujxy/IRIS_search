"""
IRIS Configuration Module

This module provides unified configuration management for the IRIS system.
It replaces the scattered config loading logic from utils/helpers.py.

Usage:
    from src.config import get_config

    # Get singleton config instance
    config = get_config()

    # Access config values
    model_path = config.models.llm_model_path
    db_path = config.storage.paper_db_path
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .loader import ConfigLoader, get_config

__all__ = [
    "get_config",
    "ConfigLoader",
    "logger",
]

logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
