"""
Milvus Service for IRIS
Direct Milvus client wrapper with metadata filtering support.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    MilvusClient = None
    DataType = None


class MilvusService:
    """
    Milvus service for vector storage and retrieval with metadata filtering.
    """

    # Default field names
    ID_FIELD = "id"
    VECTOR_FIELD = "vector"
    TEXT_FIELD = "contents"

    def __init__(
        self,
        uri: str,
        collection_name: str,
        token: Optional[str] = None,
        embedding_dim: int = 1024,
        id_field: str = ID_FIELD,
        vector_field: str = VECTOR_FIELD,
        text_field: str = TEXT_FIELD,
    ):
        """
        Initialize Milvus service.

        Args:
            uri: Milvus server URI (e.g., "http://localhost:29901")
            collection_name: Collection name
            token: Optional authentication token
            embedding_dim: Vector dimension (default: 1024 for Qwen3-Embedding-0.6B)
            id_field: ID field name (default: "id")
            vector_field: Vector field name (default: "vector")
            text_field: Text field name (default: "contents")
        """
        if MilvusClient is None:
            raise ImportError(
                "pymilvus is not installed. Install it with `pip install pymilvus`"
            )

        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.id_field = id_field
        self.vector_field = vector_field
        self.text_field = text_field

        self.client = None

        logger.info(
            f"Milvus service initialized: {uri}/{collection_name}, dim={embedding_dim}"
        )

    @staticmethod
    def _validate_collection_name(name: str) -> bool:
        """
        Validate collection name to prevent injection attacks.

        Args:
            name: Collection name to validate

        Returns:
            True if valid, False otherwise
        """
        if not name or not isinstance(name, str):
            return False
        if len(name) > 255:
            return False
        # Pattern: alphanumeric, underscore, hyphen only
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))

    def _client_connect(self) -> MilvusClient:
        """Connect to Milvus client (lazy initialization).

        Returns:
            MilvusClient instance
        """
        if self.client is None:
            if self.token:
                self.client = MilvusClient(uri=self.uri, token=self.token)
            else:
                self.client = MilvusClient(self.uri)
            logger.debug("Milvus client connected")
        return self.client

    def create_collection(
        self,
        dim: Optional[int] = None,
        overwrite: bool = False,
        collection_name: Optional[str] = None
    ) -> None:
        """
        Ensure collection exists, create if needed.

        Args:
            dim: Vector dimension (default: self.embedding_dim)
            overwrite: Whether to drop existing collection
            collection_name: Collection name (default: self.collection_name)

        Raises:
            RuntimeError: If collection creation fails
            ValueError: If collection name is invalid
        """
        collection_name = collection_name or self.collection_name
        dim = dim or self.embedding_dim

        # Validate collection name to prevent injection
        if not self._validate_collection_name(collection_name):
            raise ValueError(
                f"Invalid collection name: '{collection_name}'. "
                "Collection names must contain only alphanumeric characters, underscores, and hyphens."
            )

        client = self._client_connect()

        has_collection = client.has_collection(collection_name)

        if overwrite and has_collection:
            try:
                client.drop_collection(collection_name)
                logger.info(
                    f"Dropped existing collection: '{collection_name}'"
                )
                has_collection = False
            except Exception as e:
                logger.warning(f"Failed to drop collection: {e}")

        if has_collection:
            return

        logger.info(f"Creating collection: '{collection_name}' with dim={dim}")

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            description=f"IRIS Papers Collection: {collection_name}",
        )

        schema.add_field(
            field_name=self.id_field,
            datatype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
        )

        schema.add_field(
            field_name=self.vector_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )

        schema.add_field(
            field_name=self.text_field,
            datatype=DataType.VARCHAR,
            max_length=60000,
            description="Document content",
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=self.vector_field,
            metric_type="IP",
            index_type="AUTOINDEX",
        )

        try:
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(
                f"Successfully created collection: '{collection_name}'"
            )

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise RuntimeError(f"Milvus create collection failed: {e}") from e

    def insert(
        self,
        embeddings,
        chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
        chunk_size: int = 1000
    ) -> None:
        """
        Insert embeddings and chunks into Milvus collection.

        Args:
            embeddings: 2D numpy array of embeddings (n_chunks, dim)
            chunks: List of chunk dictionaries with metadata
            collection_name: Collection name (default: self.collection_name)
            chunk_size: Batch size for insertion
        """
        collection_name = collection_name or self.collection_name

        if not chunks:
            logger.warning("No chunks to insert")
            return

        client = self._client_connect()

        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        logger.info(f"Inserting {len(chunks)} chunks into '{collection_name}'")

        # Prepare data for insertion
        data = []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            row = {
                self.id_field: chunk.get("id", str(i)),
                self.vector_field: vec.tolist(),
                self.text_field: chunk.get("contents", ""),
            }
            # Add all metadata fields
            for key, value in chunk.items():
                if key not in [self.id_field, self.vector_field, self.text_field]:
                    row[key] = value
            data.append(row)

        # Insert in batches
        total = len(chunks)
        from tqdm import tqdm
        with tqdm(total=total, desc="Inserting to Milvus", unit="chunk") as pbar:
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                batch_data = data[start:end]

                try:
                    client.insert(collection_name=collection_name, data=batch_data)
                    pbar.update(end - start)
                except Exception as e:
                    logger.error(f"Failed to insert batch {start}-{end}: {e}")

        # Flush and load collection
        try:
            client.flush(collection_name)
            client.load_collection(collection_name)
        except Exception as e:
            logger.warning(f"Flush/load warning: {e}")

        logger.info(f"Successfully inserted {len(chunks)} chunks into '{collection_name}'")

    def search(
        self,
        query_embedding,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        collection_name: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar passages using Milvus index.

        Args:
            query_embedding: 1D numpy array of query embedding
            top_k: Number of top results to return
            filter_expr: Optional Milvus filter expression
            collection_name: Collection name (default: self.collection_name)
            output_fields: List of fields to return (default: all fields)

        Returns:
            List of chunk dictionaries with metadata

        Raises:
            ValueError: If query_embedding has invalid shape
            RuntimeError: If search fails
        """
        collection_name = collection_name or self.collection_name

        if not self._validate_collection_name(collection_name):
            raise ValueError(f"Invalid collection name: '{collection_name}'")

        client = self._client_connect()

        query_embedding = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding

        if query_embedding.ndim != 2:
            raise ValueError("Query embedding must be 1-D or 2-D array")

        # Default output fields - return all fields
        if output_fields is None:
            output_fields = ["*"]

        try:
            search_kwargs = {
                "collection_name": collection_name,
                "data": query_embedding.tolist(),
                "limit": int(top_k),
                "output_fields": output_fields,
                "consistency_level": "Bounded",
            }

            # Add filter if provided
            if filter_expr:
                search_kwargs["filter"] = filter_expr
                logger.debug(f"Using filter: {filter_expr}")

            res = client.search(**search_kwargs)

        except Exception as exc:
            logger.error(f"Search failed on '{collection_name}': {exc}")
            raise RuntimeError(f"Milvus search failed: {exc}") from exc

        # Process results - handle different pymilvus return formats
        results = []

        # Extract results from different possible pymilvus return formats
        raw_results = []
        if isinstance(res, list):
            # Direct list format (pymilvus 2.3.x+)
            raw_results = res
        elif isinstance(res, dict) and 'data' in res:
            # Dict with 'data' key format (pymilvus 2.4.x+)
            raw_results = res.get('data', [])
        elif isinstance(res, dict) and 'hits' in res:
            # HybridSearchResult format with hits
            raw_results = res.get('hits', [])
        elif isinstance(res, dict) and 'results' in res:
            # Alternative dict format
            raw_results = res.get('results', [])
        elif hasattr(res, '__iter__'):
            # Directly iterable
            raw_results = list(res)
        else:
            logger.warning(f"Unexpected search result type: {type(res)}")
            raw_results = []

        for hit in raw_results:
            # Handle both dict hits and pymilvus H class objects
            if isinstance(hit, dict):
                # Standard dictionary hit
                entity = hit
                result = {**entity}
                # Check for standard fields directly in hit
                for key in ["id", "distance", "score"]:
                    if key in hit and key not in result:
                        result[key] = hit[key]
            elif hasattr(hit, '_dict_'):
                # pymilvus H class object
                entity = hit._dict_
                result = {**entity}
                # Try to extract common fields from H class
                for key in ["id", "distance", "score"]:
                    if hasattr(hit, key):
                        result[key] = getattr(hit, key, None)
            elif hasattr(hit, '__dict__'):
                # Alternative pymilvus H class object
                entity = hit.__dict__
                result = {**entity}
                # Try to extract common fields
                for key in ["id", "distance", "score"]:
                    if hasattr(hit, key):
                        result[key] = getattr(hit, key, None)
            else:
                # Unknown hit type, try minimal extraction
                logger.debug(f"Unknown hit type: {type(hit)}, skipping")
                continue

            # Ensure required fields exist
            if 'id' not in result:
                result['id'] = entity.get('id', '')
            if 'distance' not in result:
                result['distance'] = entity.get('distance', 0.0)
            if 'score' not in result:
                result['score'] = entity.get('score', 0.0)

            results.append(result)

        logger.debug(f"Retrieved {len(results)} results from '{collection_name}'")
        return results

    def get_chunks_by_doc_id(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document ID using metadata filter.

        Args:
            doc_id: Document ID (e.g., "2401.12345")
            collection_name: Collection name (default: self.collection_name)
            limit: Maximum number of chunks to return

        Returns:
            List of chunk dictionaries
        """
        collection_name = collection_name or self.collection_name

        # Build filter expression for doc_id
        filter_expr = f'doc_id == "{doc_id}" or doc_id like "{doc_id}%"'

        # Use a dummy embedding for filter-only search
        import numpy as np
        dummy_embedding = np.zeros((1, self.embedding_dim), dtype=np.float32)

        # Search with filter
        results = self.search(
            query_embedding=dummy_embedding,
            top_k=limit,
            filter_expr=filter_expr,
            collection_name=collection_name,
        )

        logger.info(f"Retrieved {len(results)} chunks for doc_id: {doc_id}")

        return results

    def drop_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Drop a collection.

        Args:
            collection_name: Collection name (default: self.collection_name)

        Returns:
            True if successful, False if not found
        """
        collection_name = collection_name or self.collection_name

        if not self._validate_collection_name(collection_name):
            raise ValueError(f"Invalid collection name: '{collection_name}'")

        client = self._client_connect()

        if not client.has_collection(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return False

        try:
            client.drop_collection(collection_name)
            logger.info(f"Dropped collection: '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")
            return False

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collection statistics.

        Args:
            collection_name: Collection name (default: self.collection_name)

        Returns:
            Dictionary with collection stats
        """
        collection_name = collection_name or self.collection_name

        try:
            stats = self._client_connect().get_collection_stats(collection_name)
            return stats
        except Exception as e:
            logger.warning(f"Failed to get collection stats: {e}")
            return {}

    def close(self) -> None:
        """Close Milvus client."""
        if self.client is not None:
            self.client.close()
            self.client = None
        logger.debug("Milvus client closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
