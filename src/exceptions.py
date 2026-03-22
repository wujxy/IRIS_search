"""
IRIS Unified Exception Hierarchy

This module defines all custom exceptions used throughout the IRIS system.
Using specific exception types allows for better error handling and debugging.
"""


class IRISException(Exception):
    """Base exception class for all IRIS errors."""

    def __init__(self, message: str, details: dict | None = None):
        """
        Initialize IRIS exception.

        Args:
            message: Error message
            details: Additional error context
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(IRISException):
    """Raised when configuration is invalid or missing."""

    pass


class ServiceError(IRISException):
    """Raised when a service operation fails."""

    pass


class DatabaseError(IRISException):
    """Raised when database operation fails."""

    pass


class ModelNotFoundError(IRISException):
    """Raised when a required model file is not found."""

    pass


class APIError(IRISException):
    """Raised when external API call fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        details: dict | None = None,
    ):
        """
        Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code if applicable
            response_body: Response body from API
            details: Additional error context
        """
        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body


class ValidationError(IRISException):
    """Raised when input validation fails."""

    pass


class RetrievalError(IRISException):
    """Raised when document retrieval fails."""

    pass


class EmbeddingError(IRISException):
    """Raised when embedding generation fails."""

    pass


class RerankerError(IRISException):
    """Raised when reranking fails."""

    pass


class PaperProcessingError(IRISException):
    """Raised when paper processing fails."""

    pass
