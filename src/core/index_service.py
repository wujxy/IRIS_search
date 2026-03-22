"""
Index Service for IRIS
Document indexing service independent of UltraRAG.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from infrastructure.document_processor import DocumentProcessor
from infrastructure.embedding_service import EmbeddingService
from infrastructure.milvus_service import MilvusService

logger = logging.getLogger(__name__)


class IndexService:
    """
    Service for processing PDFs and building Milvus indexes.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        milvus_service: MilvusService,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_semantic_chunking: bool = False,
        semantic_model: Optional[str] = None
    ):
        """
        Initialize index service.

        Args:
            embedding_service: Embedding service instance
            milvus_service: Milvus service instance
            chunk_size: Maximum chunk size (default: 512)
            chunk_overlap: Overlap between chunks (default: 50)
            use_semantic_chunking: Use semantic chunking (default: False)
            semantic_model: Model for semantic chunking
        """
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking

        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_chunking=use_semantic_chunking,
            semantic_model=semantic_model
        )

        logger.info(
            f"IndexService initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, semantic={use_semantic_chunking}"
        )

    async def chunk_and_index(
        self,
        pdf_dir: Path,
        output_dir: Path,
        collection_name: Optional[str] = None,
        overwrite: bool = False,
        paper_id_filter: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Process PDF files and build Milvus index.

        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory for output files
            collection_name: Milvus collection name (default: milvus_service.collection_name)
            overwrite: Whether to overwrite existing collection (default: False)
            paper_id_filter: Optional filter to process only specific paper

        Returns:
            Dictionary with indexing stats
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)

        logger.info(f"Starting indexing: pdf_dir={pdf_dir}, output_dir={output_dir}")

        # Create collection
        if collection_name is None:
            collection_name = self.milvus_service.collection_name

        self.milvus_service.create_collection(
            dim=self.milvus_service.embedding_dim,
            overwrite=overwrite,
            collection_name=collection_name
        )

        # Get PDF files
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        if paper_id_filter:
            pdf_files = [f for f in pdf_files if paper_id_filter in f.name]

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {"chunk_count": 0, "paper_count": 0}

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Process all PDFs
        all_chunks = []
        paper_titles = {}

        for pdf_file in pdf_files:
            try:
                chunks = self.doc_processor.parse_and_chunk_pdf(
                    pdf_file,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )

                if chunks:
                    doc_id = chunks[0].get('doc_id', pdf_file.stem)
                    paper_titles[doc_id] = chunks[0].get('title', pdf_file.stem)
                    all_chunks.extend(chunks)
                    logger.debug(f"Processed {pdf_file.name}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue

        if not all_chunks:
            logger.warning("No chunks generated from PDFs")
            return {"chunk_count": 0, "paper_count": 0}

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        texts = [c.get('contents', '') for c in all_chunks]
        embeddings = await self.embedding_service.encode(texts)

        # Insert into Milvus
        logger.info(f"Inserting {len(all_chunks)} chunks into Milvus...")
        self.milvus_service.insert(
            embeddings=embeddings,
            chunks=all_chunks,
            collection_name=collection_name
        )

        # Save chunks file
        output_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = output_dir / "chunks.jsonl"

        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        # Build index info
        index_info = {
            "chunks_file": str(chunks_file),
            "chunk_count": len(all_chunks),
            "paper_count": len(paper_titles),
            "paper_titles": paper_titles,
            "collection_name": collection_name,
            "embedding_dim": embeddings.shape[1],
            "pdf_dir": str(pdf_dir),
        }

        logger.info(
            f"Indexing complete: {len(all_chunks)} chunks from {len(paper_titles)} papers"
        )

        # Return boolean for success check
        return bool(index_info)

    def get_index_info(self, output_dir: Path) -> Optional[Dict[str, any]]:
        """
        Get index information from output directory.

        Args:
            output_dir: Output directory path

        Returns:
            Dictionary with index info or None if not found
        """
        chunks_file = output_dir / "chunks.jsonl"

        if not chunks_file.exists():
            logger.warning(f"Index file not found: {chunks_file}")
            return None

        # Count chunks
        chunk_count = 0
        paper_ids = set()

        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk_count += 1
                    try:
                        chunk = json.loads(line)
                        paper_ids.add(chunk.get('doc_id', ''))
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error reading index file: {e}")
            return None

        return {
            "chunks_file": str(chunks_file),
            "chunk_count": chunk_count,
            "paper_count": len(paper_ids),
            "paper_ids": list(paper_ids),
        }

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension from embedding service."""
        try:
            return self.embedding_service.get_embedding_dim()
        except Exception:
            # Default dimension for Qwen3-Embedding-0.6B
            return 1024


def create_index_service_from_config(config: dict) -> IndexService:
    """
    Factory function to create IndexService from configuration.

    Args:
        config: Configuration dictionary with the following keys:
            - embedding.base_url
            - embedding.model_name
            - embedding.batch_size (optional)
            - milvus.uri
            - milvus.collection_name
            - milvus.token (optional)
            - milvus.embedding_dim
            - document.chunk_size (optional)
            - document.chunk_overlap (optional)
            - document.chunk_backend (optional)

    Returns:
        Configured IndexService instance
    """
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

    # Create Milvus service
    milvus_service = MilvusService(
        uri=config["milvus"]["uri"],
        collection_name=config["milvus"]["collection_name"],
        token=config["milvus"].get("token"),
        embedding_dim=config["milvus"]["embedding_dim"]
    )

    # Create IndexService
    document_config = config.get("document", {})
    index_service = IndexService(
        embedding_service=embedding_service,
        milvus_service=milvus_service,
        chunk_size=document_config.get("chunk_size", 512),
        chunk_overlap=document_config.get("chunk_overlap", 50),
        use_semantic_chunking=document_config.get("use_semantic", False),
        semantic_model=document_config.get("semantic_model", None)
    )

    logger.info("IndexService created from config")
    return index_service

    async def index_single_paper(
        self,
        pdf_path: Path,
        output_dir: Path,
        collection_name: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Index a single PDF file.

        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory
            collection_name: Milvus collection name

        Returns:
            Dictionary with indexing stats
        """
        pdf_dir = pdf_path.parent
        paper_id = pdf_path.stem.replace('_', '.')

        return await self.chunk_and_index(
            pdf_dir=pdf_dir,
            output_dir=output_dir,
            collection_name=collection_name,
            overwrite=False,
            paper_id_filter=paper_id
        )
