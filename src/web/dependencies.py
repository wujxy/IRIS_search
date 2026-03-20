"""
Dependency injection for IRIS Web module.
Provides FastAPI dependencies for services.
"""

from fastapi import Depends
from services.paper_service import PaperService
from utils.helpers import load_config


def get_paper_service() -> PaperService:
    """Get PaperService instance with configured database path."""
    config = load_config()
    db_path = config['storage']['paper_db_path']
    return PaperService(db_path)


def get_web_config() -> dict:
    """Get web configuration from config.yaml."""
    config = load_config()
    return config.get('web', {
        'host': '127.0.0.1',
        'port': 8000,
        'reload': True,
        'log_level': 'info',
        'pagination': {
            'default_per_page': 10,
            'max_per_page': 100
        }
    })


def get_qa_service():
    """Get QAService instance with all dependencies."""
    if not hasattr(get_qa_service, '_instance'):
        from core.qa_service import QAService
        from core.retriever import Retriever
        from infrastructure.milvus_service import MilvusService
        from infrastructure.embedding_service import EmbeddingService

        config = load_config()

        # Check if services are enabled
        if not config.get('milvus', {}).get('enabled', False):
            raise RuntimeError("Milvus service is not enabled. Please enable it in config.yaml")
        if not config.get('embedding', {}).get('enabled', False):
            raise RuntimeError("Embedding service is not enabled. Please enable it in config.yaml")

        # Create Milvus service
        milvus_service = MilvusService(
            uri=config['milvus']['uri'],
            collection_name=config['milvus']['collection_name'],
            embedding_dim=config['milvus']['embedding_dim']
        )

        # Create embedding service
        embedding_service = EmbeddingService(
            base_url=config['embedding']['base_url'],
            model_name=config['embedding']['model_name'],
            batch_size=config['embedding'].get('batch_size', 32)
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
            base_url=config['qa']['base_url'],
            model_name=config['qa']['model_name'],
            system_prompt=config['qa']['system_prompt'],
            temperature=config['qa'].get('temperature', 0.7),
            max_tokens=config['qa'].get('max_tokens', 2048),
            timeout=config['qa'].get('timeout', 120.0)
        )

        get_qa_service._instance = qa_service

    return get_qa_service._instance
