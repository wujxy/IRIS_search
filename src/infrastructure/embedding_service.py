"""
Embedding Service for IRIS
vLLM embedding model wrapper using OpenAI-compatible API.
"""

import logging
from typing import List

import numpy as np

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service using vLLM with OpenAI-compatible API.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        batch_size: int = 32,
        timeout: float = 60.0
    ):
        """
        Initialize embedding service.

        Args:
            base_url: vLLM server URL (e.g., "http://127.0.0.1:65503/v1")
            model_name: Model name for embedding
            batch_size: Batch size for encoding (default: 32)
            timeout: Request timeout in seconds (default: 60.0)
        """
        if AsyncOpenAI is None:
            raise ImportError(
                "openai is not installed. Install it with `pip install openai`"
            )

        self.base_url = base_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="dummy",
            timeout=timeout
        )

        logger.info(
            f"Embedding service initialized: {base_url}, model={model_name}, "
            f"batch_size={batch_size}"
        )

    async def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings using vLLM.

        Args:
            texts: List of text strings to encode

        Returns:
            2D numpy array of embeddings (n_texts, dim)
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

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
                batch_embeddings = [np.zeros(768, dtype=np.float32) for _ in batch]
                embeddings.extend(batch_embeddings)

        result = np.array(embeddings, dtype=np.float32)
        logger.info(f"Encoded {total} texts, shape={result.shape}")
        return result

    def encode_sync(self, texts: List[str]) -> np.ndarray:
        """
        Synchronous wrapper for encoding texts.

        Args:
            texts: List of text strings to encode

        Returns:
            2D numpy array of embeddings
        """
        import asyncio
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
        if self.client:
            self.client.close()
        logger.debug("Embedding client closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
