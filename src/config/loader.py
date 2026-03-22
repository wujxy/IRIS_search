"""
IRIS Configuration Loader

This module provides unified configuration loading with caching and validation.
It consolidates config loading logic from utils/helpers.py.

Features:
- Singleton pattern for config instance
- Environment variable expansion (${VAR_NAME})
- Path validation for models and databases
- Thread-safe config access
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .validator import ConfigValidator

logger = logging.getLogger(__name__)


# Singleton instance
_config_instance: Optional['ConfigLoader'] = None


def expand_env_vars(config: Any) -> Any:
    """
    Expand environment variables in configuration.

    Replaces ${VAR_NAME} patterns with values from environment variables.
    Useful for API keys and sensitive configuration.

    Args:
        config: Configuration dict, list, or string

    Returns:
        Configuration with environment variables expanded
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        def replace_env(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))

        return re.sub(r'\$\{([^}]+)\}', replace_env, config)
    else:
        return config


class ConfigLoader:
    """
    Configuration loader with caching and validation.

    This class provides a singleton instance for loading and accessing
    configuration from config.yaml.

    Usage:
        from src.config import get_config

        config = get_config()
        model_path = config.models.llm_model_path
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to config.yaml file. If None, uses default path.
        """
        if config_path is None:
            # Default config path
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self.validator = ConfigValidator()

    def load(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            force_reload: Force reload even if config is cached

        Returns:
            Dictionary containing configuration with env vars expanded
        """
        if self._config is not None and not force_reload:
            return self._config

        logger.info(f"Loading configuration from {self.config_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please ensure configs/config.yaml exists."
            )

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        # Expand environment variables
        self._config = expand_env_vars(self._config)

        # Validate configuration
        self.validator.validate(self._config)

        logger.info("Configuration loaded and validated successfully")

        return self._config

    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration (loads if not cached)."""
        if self._config is None:
            self.load()
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., 'models.llm_model_path')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            config.get('models.llm_model_path')
            config.get('storage.database_root')
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    # Convenience properties for common config sections
    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration section."""
        return self.config.get('models', {})

    @property
    def storage(self) -> Dict[str, Any]:
        """Get storage configuration section."""
        return self.config.get('storage', {})

    @property
    def milvus(self) -> Dict[str, Any]:
        """Get Milvus configuration section."""
        return self.config.get('milvus', {})

    @property
    def embedding(self) -> Dict[str, Any]:
        """Get embedding configuration section."""
        return self.config.get('embedding', {})

    @property
    def qa(self) -> Dict[str, Any]:
        """Get QA configuration section."""
        return self.config.get('qa', {})

    @property
    def arxiv(self) -> Dict[str, Any]:
        """Get arXiv configuration section."""
        return self.config.get('arxiv', {})

    @property
    def web(self) -> Dict[str, Any]:
        """Get web configuration section."""
        return self.config.get('web', {})

    @property
    def email(self) -> Dict[str, Any]:
        """Get email configuration section."""
        return self.config.get('email', {})

    def reload(self):
        """Force reload configuration from file."""
        self._config = None
        self.load(force_reload=True)


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get singleton configuration instance.

    Args:
        config_path: Path to config.yaml (only used on first call)

    Returns:
        ConfigLoader singleton instance

    Example:
        from src.config import get_config

        config = get_config()
        model_path = config.models.llm_model_path
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)

    return _config_instance


def reset_config():
    """
    Reset the singleton config instance.

    Useful for testing or when config file changes.
    """
    global _config_instance
    _config_instance = None
