"""
IRIS Service Base Class

This module provides a base class for all service classes in the IRIS system.
It offers common functionality like logging, error handling, and configuration management.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from src.exceptions import ServiceError


class BaseService(ABC):
    """
    Base class for all IRIS services.

    Provides common functionality:
    - Consistent logging setup
    - Error handling helpers
    - Configuration management
    - Lifecycle management
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, name: Optional[str] = None):
        """
        Initialize base service.

        Args:
            config: Service configuration dictionary
            name: Service name (defaults to class name)
        """
        self.config = config or {}
        self._name = name or self.__class__.__name__
        self._setup_logging()
        self._initialized = False

    def _setup_logging(self):
        """Set up logger for this service."""
        self.logger = logging.getLogger(self._name)

    def _log_error(self, message: str, exc_info: bool = True, **kwargs):
        """
        Log error with consistent format.

        Args:
            message: Error message
            exc_info: Include exception info (default: True)
            **kwargs: Additional context to log
        """
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.error(message, exc_info=exc_info)

    def _log_warning(self, message: str, **kwargs):
        """
        Log warning with consistent format.

        Args:
            message: Warning message
            **kwargs: Additional context to log
        """
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.warning(message)

    def _log_info(self, message: str, **kwargs):
        """
        Log info with consistent format.

        Args:
            message: Info message
            **kwargs: Additional context to log
        """
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.info(message)

    def _log_debug(self, message: str, **kwargs):
        """
        Log debug with consistent format.

        Args:
            message: Debug message
            **kwargs: Additional context to log
        """
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.debug(message)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback to default.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def initialize(self) -> None:
        """
        Initialize the service.

        Called after construction to perform any necessary setup.
        Override in subclasses for custom initialization logic.
        """
        self._log_info(f"Initializing {self._name}")
        self._initialized = True
        self._log_info(f"{self._name} initialized successfully")

    def shutdown(self) -> None:
        """
        Shutdown the service.

        Called during cleanup to release resources.
        Override in subclasses for custom shutdown logic.
        """
        self._log_info(f"Shutting down {self._name}")
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


class RetryableService(BaseService):
    """
    Base class for services that support retry logic.

    Provides automatic retry for transient failures.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize retryable service.

        Args:
            config: Service configuration
            name: Service name
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        super().__init__(config, name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
