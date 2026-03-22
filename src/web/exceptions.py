"""
Web Layer Exceptions for IRIS

Custom exceptions for the web module.
"""


class WebException(Exception):
    """Base exception for web layer."""

    pass


class ModelUnavailableError(WebException):
    """
    Raised when a model is not available for request.

    This exception can be raised when:
    - Model service is not running
    - Model failed to initialize
    - Model timeout during request
    """

    def __init__(self, model_type: str):
        """
        Initialize model unavailable error.

        Args:
            model_type: Type of model ('embedding', 'qa', 'reranker')
        """
        self.model_type = model_type
        self.message_en = f"{model_type.upper()} model is not available. Please start the model service."
        self.message_zh = f"{model_type.upper()} 模型不可用。请启动模型服务。"
        super().__init__(self.message_en)

    def get_message(self, lang: str = "en") -> str:
        """
        Get localized error message.

        Args:
            lang: Language code ('en' or 'zh')

        Returns:
            Localized error message
        """
        return self.message_zh if lang == "zh" else self.message_en


class ServiceNotReadyError(WebException):
    """Raised when a service is not ready to handle requests."""

    def __init__(self, service_name: str):
        """
        Initialize service not ready error.

        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        message = f"Service '{service_name}' is not ready. Please try again later."
        super().__init__(message)


class ConfigurationError(WebException):
    """Raised when web configuration is invalid."""

    pass


class InvalidRequestError(WebException):
    """Raised when client request is invalid."""

    def __init__(self, message: str, field: str | None = None):
        """
        Initialize invalid request error.

        Args:
            message: Error message
            field: Field that caused the error (optional)
        """
        self.field = field
        super().__init__(message)
