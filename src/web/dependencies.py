"""
Dependency injection for IRIS Web module.
Provides FastAPI dependencies for services.
"""

from fastapi import Depends
from services.paper_service import PaperService
from src.config import get_config


class ModelUnavailableError(Exception):
    """Raised when a model is not available for request."""

    def __init__(self, model_type: str):
        """
        Initialize model unavailable error.

        Args:
            model_type: Type of model ('embedding' or 'qa')
        """
        self.model_type = model_type
        self.message_en = f"{model_type.upper()} model is not available. Please start the model service."
        self.message_zh = f"{model_type.upper()} 模型不可用。请启动模型服务。"
        super().__init__(self.message_en)


def get_paper_service() -> PaperService:
    """Get PaperService instance with configured database path."""
    config = get_config()
    db_path = config.storage['paper_db_path']
    return PaperService(db_path)


def get_web_config() -> dict:
    """Get web configuration from config.yaml."""
    config = get_config()
    web_conf = config.web
    if not web_conf:
        return {
            'host': '127.0.0.1',
            'port': 8000,
            'reload': True,
            'log_level': 'info',
            'pagination': {
                'default_per_page': 10,
                'max_per_page': 100
            }
        }
    result = {
        'host': web_conf.get('host', '127.0.0.1'),
        'port': web_conf.get('port', 8000),
        'reload': web_conf.get('reload', True),
        'log_level': web_conf.get('log_level', 'info'),
        'pagination': {
            'default_per_page': 10,
            'max_per_page': 100
        }
    }
    return result


def get_qa_service():
    """Get QAService instance with all dependencies."""
    # Note: Model availability is already checked during web service startup in run_web.py
    # No need to check again on every request (was causing blocking issues)
    if not hasattr(get_qa_service, '_instance'):
        from core.qa_service import QAService
        from core.retriever import Retriever
        from infrastructure.milvus_service import MilvusService
        from infrastructure.embedding_service import EmbeddingService

        config = get_config()

        # Check if services are enabled
        if not config.milvus.get('enabled', False):
            raise RuntimeError("Milvus service is not enabled. Please enable it in config.yaml")
        if not config.embedding.get('enabled', False):
            raise RuntimeError("Embedding service is not enabled. Please enable it in config.yaml")

        # Create Milvus service
        milvus_service = MilvusService(
            uri=config.milvus['uri'],
            collection_name=config.milvus['collection_name'],
            embedding_dim=config.milvus['embedding_dim']
        )

        # Create embedding service
        embedding_service = EmbeddingService(
            base_url=config.embedding['base_url'],
            model_name=config.embedding['model_name'],
            batch_size=config.embedding.get('batch_size', 32)
        )

        # Create retriever
        retriever = Retriever(
            embedding_service=embedding_service,
            milvus_service=milvus_service,
            reranker_service=None  # Reranker is optional
        )

        # Create QA service
        qa_service = QAService(
            retriever=retriever,
            base_url=config.qa['base_url'],
            model_name=config.qa['model_name'],
            system_prompt=config.qa['system_prompt'],
            temperature=config.qa.get('temperature', 0.7),
            max_tokens=config.qa.get('max_tokens', 2048),
            timeout=config.qa.get('timeout', 120.0)
        )

        get_qa_service._instance = qa_service

    return get_qa_service._instance
