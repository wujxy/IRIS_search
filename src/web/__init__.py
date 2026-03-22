"""
IRIS Web Module
Web interface for literature browsing and search.
"""

__version__ = "1.0.0"

from .loaders import (
    create_paper_service,
    create_qa_service,
    create_retriever,
    get_cached_paper_service,
    get_cached_qa_service,
    get_web_config,
    clear_service_cache,
)

from .exceptions import (
    WebException,
    ModelUnavailableError,
    ServiceNotReadyError,
    ConfigurationError as WebConfigurationError,
    InvalidRequestError,
)

__all__ = [
    # Loaders
    "create_paper_service",
    "create_qa_service",
    "create_retriever",
    "get_cached_paper_service",
    "get_cached_qa_service",
    "get_web_config",
    "clear_service_cache",
    # Exceptions
    "WebException",
    "ModelUnavailableError",
    "ServiceNotReadyError",
    "WebConfigurationError",
    "InvalidRequestError",
]
