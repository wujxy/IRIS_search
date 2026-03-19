"""
Core Package for IRIS
检索核心逻辑 (Retriever, IndexService, QAService)
"""

from .retriever import Retriever
from .index_service import IndexService
from .qa_service import QAService
from .prompt_templates import PromptTemplate

__all__ = [
    "Retriever",
    "IndexService",
    "QAService",
    "PromptTemplate",
]
