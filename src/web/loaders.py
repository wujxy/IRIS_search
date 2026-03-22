"""
Service Loaders for IRIS Web Module

This module provides factory functions for creating service instances.
It isolates the service creation logic from FastAPI dependencies.
"""

import logging
from typing import Any, Callable, Optional

from src.config import get_config
from src.exceptions import ConfigurationError

# Type aliases for services (avoiding circular imports)
PaperService = Any  # services.paper_service.PaperService
MilvusService = Any  # infrastructure.milvus_service.MilvusService
EmbeddingService = Any  # infrastructure.embedding_service.EmbeddingService
RerankerService = Any  # infrastructure.reranker_service.RerankerService
Retriever = Any  # core.retriever.Retriever
QAService = Any  # core.qa_service.QAService

logger = logging.getLogger(__name__)


# Service singleton cache
_service_cache: dict[str, Any] = {}


def create_paper_service() -> PaperService:
    """
    Create PaperService instance with configured database path.

    Returns:
        PaperService instance
    """
    from services.paper_service import PaperService

    config = get_config()
    db_path = config.storage.get('paper_db_path')
    if not db_path:
        raise ConfigurationError("paper_db_path not configured")

    return PaperService(db_path)


def create_milvus_service() -> MilvusService:
    """
    Create MilvusService instance from configuration.

    Returns:
        MilvusService instance

    Raises:
        ConfigurationError: If Milvus is not properly configured
    """
    from infrastructure.milvus_service import MilvusService

    config = get_config()
    milvus_config = config.milvus

    if not milvus_config.get('enabled', False):
        raise ConfigurationError("Milvus service is not enabled")

    return MilvusService(
        uri=milvus_config['uri'],
        collection_name=milvus_config['collection_name'],
        embedding_dim=milvus_config.get('embedding_dim', 1024)
    )


def create_embedding_service() -> EmbeddingService:
    """
    Create EmbeddingService instance from configuration.

    Returns:
        EmbeddingService instance

    Raises:
        ConfigurationError: If embedding service is not properly configured
    """
    from infrastructure.embedding_service import EmbeddingService

    config = get_config()
    embedding_config = config.embedding

    if not embedding_config.get('enabled', False):
        raise ConfigurationError("Embedding service is not enabled")

    return EmbeddingService(
        base_url=embedding_config['base_url'],
        model_name=embedding_config['model_name'],
        batch_size=embedding_config.get('batch_size', 32)
    )


def create_reranker_service() -> Optional[RerankerService]:
    """
    Create RerankerService instance from configuration.

    Returns:
        RerankerService instance or None if disabled
    """
    from infrastructure.reranker_service import RerankerService

    config = get_config()
    reranker_config = config.get('reranker', {})

    if not reranker_config.get('enabled', False):
        return None

    return RerankerService(
        model_path=reranker_config['model_path'],
        batch_size=reranker_config.get('batch_size', 16),
        device=reranker_config.get('device', 'cpu')
    )


def create_retriever() -> Retriever:
    """
    Create Retriever instance with all required services.

    Returns:
        Retriever instance

    Raises:
        ConfigurationError: If required services are not configured
    """
    from core.retriever import Retriever

    milvus_service = create_milvus_service()
    embedding_service = create_embedding_service()
    reranker_service = create_reranker_service()

    return Retriever(
        embedding_service=embedding_service,
        milvus_service=milvus_service,
        reranker_service=reranker_service
    )


def create_qa_service() -> QAService:
    """
    Create QAService instance with all dependencies.

    Returns:
        QAService instance

    Raises:
        ConfigurationError: If required services are not configured
    """
    from core.qa_service import QAService

    config = get_config()
    qa_config = config.qa

    retriever = create_retriever()

    return QAService(
        retriever=retriever,
        base_url=qa_config['base_url'],
        model_name=qa_config['model_name'],
        system_prompt=qa_config.get('system_prompt', 'You are a professional research assistant.'),
        temperature=qa_config.get('temperature', 0.7),
        max_tokens=qa_config.get('max_tokens', 2048),
        timeout=qa_config.get('timeout', 120.0)
    )


def get_or_create_service(service_name: str, factory_func: Callable[[], Any]) -> Any:
    """
    Get cached service instance or create new one.

    Args:
        service_name: Name for caching
        factory_func: Factory function to create service

    Returns:
        Service instance
    """
    if service_name not in _service_cache:
        _service_cache[service_name] = factory_func()
        logger.debug(f"Created new service instance: {service_name}")

    return _service_cache[service_name]


def clear_service_cache() -> None:
    """Clear all cached service instances."""
    _service_cache.clear()
    logger.debug("Service cache cleared")


def get_cached_paper_service() -> PaperService:
    """Get cached PaperService instance."""
    return get_or_create_service('paper_service', create_paper_service)


def get_cached_qa_service() -> QAService:
    """Get cached QAService instance."""
    return get_or_create_service('qa_service', create_qa_service)


def get_web_config() -> dict[str, Any]:
    """
    Get web configuration from config.yaml.

    Returns:
        Dictionary with web configuration
    """
    config = get_config()
    web_conf = config.web

    defaults = {
        'host': '127.0.0.1',
        'port': 8000,
        'reload': True,
        'log_level': 'info',
        'pagination': {
            'default_per_page': 10,
            'max_per_page': 100
        }
    }

    if not web_conf:
        return defaults

    return {
        'host': web_conf.get('host', defaults['host']),
        'port': web_conf.get('port', defaults['port']),
        'reload': web_conf.get('reload', defaults['reload']),
        'log_level': web_conf.get('log_level', defaults['log_level']),
        'pagination': defaults['pagination']
    }
