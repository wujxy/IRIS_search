"""
Embedding Service for IRIS
Supports local vLLM and external API providers (OpenAI, Cohere, etc.).
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

import numpy as np

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# Optional Cohere support
try:
    import cohere
except ImportError:
    cohere = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Universal embedding service supporting local vLLM and external APIs.

    Supported providers:
    - local: vLLM server (default)
    - openai: OpenAI embeddings API
    - cohere: Cohere embeddings API
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        provider: str = "local",
        api_key: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        timeout: float = 60.0
    ):
        """
        Initialize embedding service with provider detection.

        Args:
            base_url: API base URL (for local vLLM or custom endpoints)
            model_name: Model name for embedding
            provider: Provider type (local, openai, cohere, azure_openai)
            api_key: API key for external providers
            provider_config: Provider-specific configuration
            batch_size: Batch size for encoding (default: 32)
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.provider = provider or "local"
        self.provider_config = provider_config or {}
        self.base_url = base_url
        self.batch_size = batch_size
        self.timeout = timeout

        # Validate and initialize provider
        if self.provider == "local":
            self._init_local_client(base_url, model_name, timeout)
        elif self.provider == "openai":
            self._init_openai_client(base_url, model_name, api_key, timeout)
        elif self.provider == "cohere":
            self._init_cohere_client(api_key, model_name)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.model_name = model_name

    def _init_local_client(self, base_url: str, model_name: str, timeout: float):
        """Initialize local vLLM client (existing implementation)."""
        if AsyncOpenAI is None:
            raise ImportError(
                "openai is not installed. Install it with `pip install openai`"
            )

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="dummy",  # vLLM doesn't require real key
            timeout=timeout
        )
        self._encode_method = self._encode_vllm

        logger.info(
            f"Embedding service initialized (local vLLM): {base_url}, model={model_name}"
        )

    def _init_openai_client(self, base_url: str, model_name: str, api_key: Optional[str], timeout: float):
        """Initialize OpenAI client for embeddings."""
        if AsyncOpenAI is None:
            raise ImportError(
                "openai is not installed. Install it with `pip install openai`"
            )
        if not api_key:
            raise ValueError("api_key is required for openai provider")

        self.client = AsyncOpenAI(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            timeout=timeout
        )
        self._encode_method = self._encode_openai
        # Use provider-specific model name if configured
        openai_config = self.provider_config.get("openai", {})
        self.model_name = openai_config.get("model", model_name)

        logger.info(
            f"Embedding service initialized (OpenAI): model={self.model_name}"
        )

    def _init_cohere_client(self, api_key: Optional[str], model_name: str):
        """Initialize Cohere client for embeddings."""
        if cohere is None:
            raise ImportError(
                "cohere is not installed. Install it with `pip install cohere`"
            )
        if not api_key:
            raise ValueError("api_key is required for cohere provider")

        self.client = cohere.AsyncClient(api_key)
        self._encode_method = self._encode_cohere

        # Use provider-specific model name if configured
        cohere_config = self.provider_config.get("cohere", {})
        self.model_name = cohere_config.get("model", model_name)
        self.input_type = cohere_config.get("input_type", "search_document")

        logger.info(
            f"Embedding service initialized (Cohere): model={self.model_name}"
        )

    async def _encode_vllm(self, texts: List[str]) -> np.ndarray:
        """Encode using local vLLM (existing implementation)."""
        embeddings = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Encoded batch {i//self.batch_size + 1}: {len(batch)} texts")

            except Exception as e:
                logger.error(f"Failed to encode batch {i}-{i+self.batch_size}: {e}")
                # Fill with zeros on error
                batch_embeddings = [np.zeros(1024, dtype=np.float32) for _ in batch]
                embeddings.extend(batch_embeddings)

        return np.array(embeddings, dtype=np.float32)

    async def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """Encode using OpenAI API."""
        embeddings = []
        total = len(texts)

        # OpenAI has different batch limits (max 2048 texts for v3)
        openai_batch_size = min(self.batch_size, 2048)

        for i in range(0, total, openai_batch_size):
            batch = texts[i:i + openai_batch_size]

            try:
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Encoded batch {i//openai_batch_size + 1}: {len(batch)} texts")

            except Exception as e:
                logger.error(f"Failed to encode batch {i}-{i+openai_batch_size}: {e}")
                # Estimate OpenAI embedding dimension (1536 for text-embedding-3-small)
                dim = self.provider_config.get("openai", {}).get("dimensions", 1536)
                batch_embeddings = [np.zeros(dim, dtype=np.float32) for _ in batch]
                embeddings.extend(batch_embeddings)

        return np.array(embeddings, dtype=np.float32)

    async def _encode_cohere(self, texts: List[str]) -> np.ndarray:
        """Encode using Cohere API."""
        embeddings = []
        total = len(texts)

        # Cohere has batch limit of 96 texts
        cohere_batch_size = min(self.batch_size, 96)

        for i in range(0, total, cohere_batch_size):
            batch = texts[i:i + cohere_batch_size]

            try:
                response = await self.client.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type=self.input_type
                )
                batch_embeddings = response.embeddings
                embeddings.extend(batch_embeddings)
                logger.debug(f"Encoded batch {i//cohere_batch_size + 1}: {len(batch)} texts")

            except Exception as e:
                logger.error(f"Failed to encode batch {i}-{i+cohere_batch_size}: {e}")
                # Estimate Cohere embedding dimension (1024 for embed-english-v3.0)
                batch_embeddings = [np.zeros(1024, dtype=np.float32) for _ in batch]
                embeddings.extend(batch_embeddings)

        return np.array(embeddings, dtype=np.float32)

    async def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings using configured provider.

        Args:
            texts: List of text strings to encode

        Returns:
            2D numpy array of embeddings (n_texts, dim)
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        result = await self._encode_method(texts)
        logger.info(f"Encoded {len(texts)} texts with {self.provider}, shape={result.shape}")
        return result

    def encode_sync(self, texts: List[str]) -> np.ndarray:
        """
        Synchronous wrapper for encoding texts.

        Args:
            texts: List of text strings to encode

        Returns:
            2D numpy array of embeddings
        """
        return asyncio.run(self.encode(texts))

    def get_embedding_dim(self, sample_text: str = "sample") -> int:
        """
        Get embedding dimension by encoding a sample text.

        Args:
            sample_text: Sample text to encode

        Returns:
            Embedding dimension
        """
        embedding = self.encode_sync([sample_text])
        return embedding.shape[1]

    def close(self) -> None:
        """Close the client connection."""
        if hasattr(self, 'client') and self.client:
            if self.provider == "cohere":
                # Cohere client doesn't have close method
                pass
            else:
                self.client.close()
        logger.debug("Embedding client closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
