"""
Reranker Service for IRIS
Supports local CrossEncoder and external API providers (Cohere, etc.).
"""

import asyncio
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# Optional Cohere support
try:
    import cohere
except ImportError:
    cohere = None

# Optional OpenAI client for custom endpoints
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Universal reranker service supporting local CrossEncoder and external APIs.

    Supported providers:
    - local: sentence-transformers CrossEncoder (default)
    - cohere: Cohere rerank API
    - openai_compatible: OpenAI-compatible rerank endpoints
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 16,
        device: str = "cpu",
        provider: str = "local",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize reranker service with provider detection.

        Args:
            model_path: Path to local CrossEncoder model (required for local provider)
            batch_size: Batch size for prediction (default: 16)
            device: Device to use for local model (default: "cpu")
            provider: Provider type (local, cohere, openai_compatible)
            api_key: API key for external providers
            api_url: Custom API URL for OpenAI-compatible endpoints
            provider_config: Provider-specific configuration
        """
        self.provider = provider or "local"
        self.provider_config = provider_config or {}
        self.batch_size = batch_size

        # Validate and initialize provider
        if self.provider == "local":
            if not model_path:
                raise ValueError("model_path is required for local provider")
            self._init_local_model(model_path, device)
        elif self.provider == "cohere":
            self._init_cohere_client(api_key)
        elif self.provider == "openai_compatible":
            self._init_openai_compatible_client(api_url, api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _init_local_model(self, model_path: str, device: str):
        """Initialize local CrossEncoder (existing implementation)."""
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with `pip install sentence-transformers`"
            )

        self.model_path = model_path
        self.device = device

        try:
            self.model = CrossEncoder(model_path, device=device)
            self._rerank_method = self._rerank_local
            logger.info(
                f"Reranker initialized (local): {model_path}, device={device}"
            )
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    def _init_cohere_client(self, api_key: Optional[str]):
        """Initialize Cohere rerank client."""
        if cohere is None:
            raise ImportError(
                "cohere is not installed. Install it with `pip install cohere`"
            )
        if not api_key:
            raise ValueError("api_key is required for cohere provider")

        self.client = cohere.AsyncClient(api_key)
        self._rerank_method = self._rerank_cohere_async

        cohere_config = self.provider_config.get("cohere", {})
        self.model_name = cohere_config.get("model", "rerank-english-v3.0")
        self.default_top_n = cohere_config.get("top_n", None)

        logger.info(f"Reranker initialized (Cohere): model={self.model_name}")

    def _init_openai_compatible_client(self, api_url: Optional[str], api_key: Optional[str]):
        """Initialize OpenAI-compatible rerank client."""
        if AsyncOpenAI is None:
            raise ImportError(
                "openai is not installed. Install it with `pip install openai`"
            )
        if not api_url:
            raise ValueError("api_url is required for openai_compatible provider")

        openai_config = self.provider_config.get("openai_compatible", {})
        self.client = AsyncOpenAI(
            base_url=api_url,
            api_key=api_key or "dummy"
        )
        self.model_name = openai_config.get("model", "reranker-default")
        self._rerank_method = self._rerank_openai_compatible_async

        logger.info(f"Reranker initialized (OpenAI-compatible): endpoint={api_url}")

    def _rerank_local(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank using local CrossEncoder (existing implementation)."""
        if not passages:
            logger.warning("No passages to rerank")
            return []

        if top_k is None:
            top_k = len(passages)
        elif top_k > len(passages):
            top_k = len(passages)

        try:
            # Construct query-passage pairs
            pairs = [[query, passage] for passage in passages]

            # Predict scores
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # Convert to numpy if tensor
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            scores = scores.flatten().astype(float)

            # Get top_k indices sorted by score descending
            top_indices = np.argsort(scores)[-top_k:][::-1]

            # Return (index, score) tuples
            result = [(int(idx), float(scores[idx])) for idx in top_indices]

            logger.debug(f"Reranked {len(passages)} passages, returning top {len(result)}")
            return result

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order on error
            return [(i, 1.0) for i in range(min(top_k or len(passages), len(passages)))]

    async def _rerank_cohere_async(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank using Cohere API (async)."""
        if not passages:
            logger.warning("No passages to rerank")
            return []

        if top_k is None:
            top_k = self.default_top_n or len(passages)

        try:
            response = await self.client.rerank(
                model=self.model_name,
                query=query,
                documents=[{"text": p} for p in passages],
                top_n=top_k,
                return_documents=False
            )

            # Convert Cohere response to (index, score) tuples
            results = [(result.index, result.relevance_score)
                       for result in response.results]

            logger.debug(f"Reranked {len(passages)} passages via Cohere, returning top {len(results)}")
            return results

        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # Return original order on error
            return [(i, 1.0) for i in range(min(top_k or len(passages), len(passages)))]

    async def _rerank_openai_compatible_async(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank using OpenAI-compatible endpoint (async)."""
        if not passages:
            logger.warning("No passages to rerank")
            return []

        if top_k is None:
            top_k = len(passages)

        try:
            # OpenAI-compatible rerank API format
            # Note: This is a common pattern, actual implementation may vary
            response = await self.client.post(
                "/rerank",
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": passages,
                    "top_k": top_k
                }
            )

            data = response.json()
            # Assuming response format: {"results": [{"index": int, "score": float}, ...]}
            results = [(item["index"], item["score"]) for item in data.get("results", [])]

            logger.debug(f"Reranked {len(passages)} passages via OpenAI-compatible, returning top {len(results)}")
            return results

        except Exception as e:
            logger.error(f"OpenAI-compatible reranking failed: {e}")
            # Return original order on error
            return [(i, 1.0) for i in range(min(top_k or len(passages), len(passages)))]

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank passages based on query relevance using configured provider.

        Args:
            query: Query string
            passages: List of passage texts
            top_k: Number of top results to return (default: len(passages))

        Returns:
            List of (index, score) tuples sorted by score descending

        Raises:
            ValueError: If passages is empty
        """
        if not passages:
            logger.warning("No passages to rerank")
            return []

        if top_k is None:
            top_k = len(passages)
        elif top_k > len(passages):
            top_k = len(passages)

        # Check if async method
        if asyncio.iscoroutinefunction(self._rerank_method):
            # Run async method synchronously
            return asyncio.run(self._rerank_method(query, passages, top_k))
        else:
            # Run synchronous method
            return self._rerank_method(query, passages, top_k)

    def rerank_with_metadata(
        self,
        query: str,
        results: List[Dict],
        text_field: str = "contents",
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank results with metadata preserved.

        Args:
            query: Query string
            results: List of result dictionaries
            text_field: Field name containing text (default: "contents")
            top_k: Number of top results to return

        Returns:
            Reranked list of result dictionaries
        """
        if not results:
            return []

        # Extract passages for scoring
        passages = [result.get(text_field, "") for result in results]

        # Rerank
        reranked_indices = self.rerank(query, passages, top_k=top_k)

        # Return reordered results with original metadata
        return [results[idx] for idx, _ in reranked_indices]

    def compute_scores(
        self,
        query: str,
        passages: List[str]
    ) -> List[float]:
        """
        Compute relevance scores for query-passage pairs.

        Args:
            query: Query string
            passages: List of passage texts

        Returns:
            List of relevance scores
        """
        if self.provider == "local":
            pairs = [[query, passage] for passage in passages]
            scores = self.model.predict(pairs, batch_size=self.batch_size)

            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            scores = scores.flatten().tolist()

            return scores
        else:
            # For external providers, use rerank and extract scores
            reranked = self.rerank(query, passages, top_k=len(passages))
            # Reconstruct full score list
            score_dict = {idx: score for idx, score in reranked}
            return [score_dict.get(i, 0.0) for i in range(len(passages))]
