"""
Reranker Service for IRIS
Cross-encoder based reranking using sentence-transformers.
"""

import logging
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Reranker service using CrossEncoder for query-passage scoring.
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int = 16,
        device: str = "cuda:0"
    ):
        """
        Initialize reranker service.

        Args:
            model_path: Path to CrossEncoder model
            batch_size: Batch size for prediction (default: 16)
            device: Device to use (default: "cuda:0")
        """
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with `pip install sentence-transformers`"
            )

        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device

        try:
            self.model = CrossEncoder(model_path, device=device)
            logger.info(
                f"Reranker loaded: {model_path}, device={device}, "
                f"batch_size={batch_size}"
            )
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank passages based on query relevance.

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
        pairs = [[query, passage] for passage in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
        scores = scores.flatten().tolist()

        return scores
