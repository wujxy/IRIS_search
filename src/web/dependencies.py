"""
FastAPI Dependencies for IRIS Web Module

This module provides FastAPI dependency functions for route handlers.
It uses the loaders module for service creation and caching.
"""

from fastapi import Depends

from src.web.loaders import (
    get_cached_paper_service,
    get_cached_qa_service,
    get_web_config,
)
from src.web.exceptions import ModelUnavailableError


def get_paper_service():
    """
    FastAPI dependency for PaperService.

    Returns:
        PaperService instance (cached)

    Example:
        @app.get("/papers")
        def list_papers(service: PaperService = Depends(get_paper_service)):
            return service.list_papers()
    """
    return get_cached_paper_service()


def get_qa_service():
    """
    FastAPI dependency for QAService.

    Returns:
        QAService instance (cached)

    Note:
        Model availability is checked during web service startup
        in run_web.py. No need to check on every request.

    Example:
        @app.post("/qa")
        async def ask_question(
            question: str,
            service: QAService = Depends(get_qa_service)
        ):
            return await service.query(question)
    """
    return get_cached_qa_service()


def get_web_config_dependency():
    """
    FastAPI dependency for web configuration.

    Returns:
        Web configuration dictionary

    Example:
        @app.get("/config")
        def get_config(config: dict = Depends(get_web_config_dependency)):
            return config
    """
    return get_web_config()


# Export exceptions for use in routes
__all__ = [
    "get_paper_service",
    "get_qa_service",
    "get_web_config_dependency",
    "ModelUnavailableError",
]
