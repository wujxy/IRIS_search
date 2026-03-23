"""
IRIS Configuration Models

Pydantic models for type-safe configuration access.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ModelsConfig(BaseModel):
    """Models configuration."""
    embedding_model_path: str
    reranker_model_path: str
    llm_model_path: str

    @field_validator('embedding_model_path', 'reranker_model_path', 'llm_model_path')
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate model path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model path not found: {v}")
        return v


class StorageConfig(BaseModel):
    """Storage configuration."""
    database_root: str
    paper_db_path: str


class MilvusConfig(BaseModel):
    """Milvus configuration."""
    uri: str = "http://localhost:29901"
    collection_name: str = "iris_papers"
    master_collection: str = "iris_master"
    embedding_dim: int = 1024
    enabled: bool = True


class EmbeddingConfig(BaseModel):
    """Embedding service configuration."""
    base_url: str
    model_name: str
    batch_size: int = 32
    enabled: bool = True
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.15
    tensor_parallel_size: int = 1
    enforce_eager: bool = True


class RerankerConfig(BaseModel):
    """Reranker configuration."""
    model_path: str
    batch_size: int = 16
    device: str = "cpu"
    enabled: bool = False


class QAConfig(BaseModel):
    """QA service configuration."""
    model_name: str
    question_set_path: str = "./configs/questions.txt"
    system_prompt: str = "You are a professional research assistant."
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 2048
    base_url: str = "http://127.0.0.1:65504/v1"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    enforce_eager: bool = True


class DocumentConfig(BaseModel):
    """Document processing configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    use_title: bool = True
    use_semantic: bool = False
    chunk_backend: str = "sentence"
    remove_references: bool = True
    reference_keywords: List[str] = [
        "References", "Bibliography",
        "REFERENCES", "BIBLIOGRAPHY",
        "参考文献"
    ]


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = 5
    rerank_multiplier: int = 3


class ArxivConfig(BaseModel):
    """arXiv configuration."""
    keywords: List[str]
    max_results_per_keyword: int = 20
    sort_by: str = "SubmittedDate"


class FilteringConfig(BaseModel):
    """Filtering configuration."""
    exclude_reviews: bool = True
    review_keywords: List[str] = ["review", "survey", "overview"]


class EmailConfig(BaseModel):
    """Email configuration."""
    enabled: bool = False
    sender: Optional[str] = None
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    password: Optional[str] = None
    receiver: Optional[str] = None


class WebConfig(BaseModel):
    """Web configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"


class UpdateConfig(BaseModel):
    """Update configuration."""
    interval_hours: int = 2
