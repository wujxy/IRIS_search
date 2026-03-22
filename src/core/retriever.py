"""
Retriever for IRIS
Integrates embedding, Milvus search, and reranking.
"""

import asyncio
import logging
from typing import Dict, List, Optional

import numpy as np

from infrastructure.embedding_service import EmbeddingService
from infrastructure.milvus_service import MilvusService
from infrastructure.reranker_service import RerankerService

logger = logging.getLogger(__name__)


class Retriever:
    """
    Unified retriever combining embedding, vector search, and reranking.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        milvus_service: MilvusService,
        reranker_service: Optional[RerankerService] = None,
        default_top_k: int = 5,
        retrieval_top_k: int = 15  # Retrieve more for reranking
    ):
        """
        Initialize retriever.

        Args:
            embedding_service: Embedding service instance
            milvus_service: Milvus service instance
            reranker_service: Optional reranker service
            default_top_k: Default number of results to return (default: 5)
            retrieval_top_k: Number to retrieve before reranking (default: 15)
        """
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.reranker_service = reranker_service
        self.default_top_k = default_top_k
        self.retrieval_top_k = retrieval_top_k

        logger.info(
            f"Retriever initialized: default_top_k={default_top_k}, "
            f"retrieval_top_k={retrieval_top_k}, "
            f"reranker_enabled={reranker_service is not None}"
        )

    async def retrieve(
        self,
        query: str,
        mode: str = "global",
        paper_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query string
            mode: Query mode ("global" or "specific")
            paper_id: Paper ID for specific mode filtering
            top_k: Number of results to return (default: self.default_top_k)

        Returns:
            List of retrieved chunk dictionaries with metadata

        Raises:
            ValueError: If mode is invalid
        """
        if top_k is None:
            top_k = self.default_top_k

        logger.info(f"Retrieving for query: '{query}', mode={mode}, top_k={top_k}")

        # Validate mode
        if mode not in ["global", "specific"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'global' or 'specific'")

        # Build filter expression
        filter_expr = self._build_filter(mode, paper_id)

        # 1. Generate query embedding
        query_emb = await self.embedding_service.encode([query])

        if query_emb.shape[0] == 0:
            logger.warning("Failed to generate query embedding")
            return []

        # 2. Milvus search (retrieve more if reranking)
        retrieval_count = min(self.retrieval_top_k, top_k * 3) if self.reranker_service else top_k

        logger.debug(f"Milvus search: top_k={retrieval_count}, filter={filter_expr}")

        # Run synchronous Milvus search in thread pool to avoid blocking event loop
        results = await asyncio.to_thread(
            self.milvus_service.search,
            query_emb[0],
            top_k=retrieval_count,
            filter_expr=filter_expr,
            output_fields=["*"]  # Explicitly request all fields
        )

        logger.debug(f"Retrieved {len(results)} chunks from Milvus")
        if results:
            logger.debug(f"First result keys: {list(results[0].keys())}")
            logger.debug(f"First result has 'contents': {'contents' in results[0]}")

        # 3. Rerank if enabled and have enough results
        if self.reranker_service and len(results) > top_k:
            results = self._rerank_results(query, results, top_k)

        # Return top_k results
        final_results = results[:top_k]
        logger.info(f"Returning {len(final_results)} results")

        return final_results

    def retrieve_sync(
        self,
        query: str,
        mode: str = "global",
        paper_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Synchronous wrapper for retrieve.

        Args:
            query: Query string
            mode: Query mode ("global" or "specific")
            paper_id: Paper ID for specific mode filtering
            top_k: Number of results to return

        Returns:
            List of retrieved chunk dictionaries
        """
        import asyncio
        return asyncio.run(self.retrieve(query, mode, paper_id, top_k))

    def _build_filter(self, mode: str, paper_id: Optional[str]) -> Optional[str]:
        """
        Build Milvus filter expression based on mode and paper_id.

        Args:
            mode: Query mode
            paper_id: Paper ID for specific mode

        Returns:
            Filter expression string or None
        """
        if mode == "specific" and paper_id:
            # Match exact paper_id or paper_id with version suffix
            return f'doc_id == "{paper_id}" or doc_id like "{paper_id}%"'
        return None

    def _rerank_results(
        self,
        query: str,
        results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Rerank results using CrossEncoder.

        Args:
            query: Query string
            results: Retrieved results
            top_k: Number of top results to return

        Returns:
            Reranked results list
        """
        try:
            reranked = self.reranker_service.rerank_with_metadata(
                query=query,
                results=results,
                text_field="contents",
                top_k=top_k
            )
            logger.debug(f"Reranked {len(results)} to {len(reranked)} results")
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original results")
            return results[:top_k]

    async def retrieve_batch(
        self,
        queries: List[str],
        mode: str = "global",
        paper_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, any]]]:
        """
        Batch retrieval for multiple queries.

        Args:
            queries: List of query strings
            mode: Query mode
            paper_id: Paper ID for specific mode
            top_k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        results = []
        for query in queries:
            query_results = await self.retrieve(query, mode, paper_id, top_k)
            results.append(query_results)
        return results

    def get_chunks_by_doc_id(
        self,
        doc_id: str,
        limit: int = 1000
    ) -> List[Dict[str, any]]:
        """
        Get all chunks for a specific document ID.

        Args:
            doc_id: Document ID (e.g., "2401.12345")
            limit: Maximum number of chunks to return

        Returns:
            List of chunk dictionaries
        """
        return self.milvus_service.get_chunks_by_doc_id(doc_id, limit=limit)


def create_retriever_from_config(config: dict) -> Retriever:
    """
    Factory function to create Retriever from configuration.

    Args:
        config: Configuration dictionary with the following keys:
            - milvus.uri
            - milvus.collection_name
            - milvus.embedding_dim
            - milvus.token (optional)
            - embedding.base_url
            - embedding.model_name
            - embedding.batch_size (optional)
            - reranker.enabled (optional)
            - models.reranker_model_path
            - reranker.device (optional)
            - reranker.batch_size (optional)
            - retrieval.top_k (optional)
            - retrieval.rerank_multiplier (optional)

    Returns:
        Configured Retriever instance
    """
    # Create Milvus service
    milvus_service = MilvusService(
        uri=config["milvus"]["uri"],
        collection_name=config["milvus"]["collection_name"],
        token=config["milvus"].get("token"),
        embedding_dim=config["milvus"]["embedding_dim"]
    )

    # Create Embedding service
    embedding_config = config["embedding"]
    embedding_service = EmbeddingService(
        base_url=embedding_config["base_url"],
        model_name=embedding_config["model_name"],
        provider=embedding_config.get("provider", "local"),
        api_key=embedding_config.get("api_key"),
        provider_config=embedding_config,
        batch_size=embedding_config.get("batch_size", 32)
    )

    # Create Reranker service (optional)
    reranker_service = None
    if config.get("reranker", {}).get("enabled", False):
        from infrastructure.reranker_service import RerankerService
        reranker_config = config["reranker"]
        reranker_service = RerankerService(
            model_path=config.get("models", {}).get("reranker_model_path"),
            device=reranker_config.get("device", "cpu"),
            batch_size=reranker_config.get("batch_size", 16),
            provider=reranker_config.get("provider", "local"),
            api_key=reranker_config.get("api_key"),
            api_url=reranker_config.get("api_url"),
            provider_config=reranker_config
        )

    # Create Retriever
    retriever = Retriever(
        embedding_service=embedding_service,
        milvus_service=milvus_service,
        reranker_service=reranker_service,
        default_top_k=config.get("retrieval", {}).get("top_k", 5),
        retrieval_top_k=config.get("retrieval", {}).get("rerank_multiplier", 3) * 5
    )

    logger.info("Retriever created from config")
    return retriever
