"""
Document Processor for IRIS
PDF parsing and text chunking functionality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    from chonkie import SemanticChunker
except ImportError:
    SemanticChunker = None

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document processor for PDF parsing and text chunking.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_semantic_chunking: bool = False,
        semantic_model: Optional[str] = None
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Maximum chunk size in tokens (default: 512)
            chunk_overlap: Overlap between chunks (default: 50)
            use_semantic_chunking: Use semantic chunking (default: False)
            semantic_model: Model for semantic chunking (optional)
        """
        if fitz is None:
            raise ImportError(
                "pymupdf is not installed. Install it with `pip install pymupdf`"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking

        # Initialize semantic chunker if requested
        self.semantic_chunker = None
        if use_semantic_chunking and semantic_model:
            if SemanticChunker is None:
                logger.warning("chonkie not installed, falling back to simple chunking")
                self.use_semantic_chunking = False
            else:
                self.semantic_chunker = SemanticChunker(
                    model_or_path=semantic_model,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )
                logger.info(f"Semantic chunking enabled with model: {semantic_model}")

        logger.info(
            f"Document processor initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, semantic={use_semantic_chunking}"
        )

    def parse_pdf(self, pdf_path: Path) -> Dict[str, str]:
        """
        Parse PDF file and extract text and metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with 'id', 'title', and 'contents'

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF parsing fails
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))

            # Extract title from filename or first page
            title = pdf_path.stem
            try:
                first_page = doc[0]
                if first_page.get_text():
                    # Try to extract title from first page (simple heuristic)
                    lines = first_page.get_text().split('\n')
                    for line in lines[:5]:
                        if len(line.strip()) > 10 and len(line.strip()) < 100:
                            title = line.strip()
                            break
            except Exception:
                pass

            # Extract all text
            contents = ""
            for page in doc:
                page_text = page.get_text()
                contents += page_text + "\n"

            doc.close()

            # Extract paper ID from filename
            paper_id = pdf_path.stem.replace('_', '.')

            logger.debug(f"Parsed PDF: {pdf_path.name}, pages={len(doc)}, chars={len(contents)}")

            return {
                "id": paper_id,
                "title": title,
                "contents": contents.strip()
            }

        except Exception as e:
            logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            raise ValueError(f"PDF parsing failed: {e}") from e

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        title: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into manageable pieces.

        Args:
            text: Text to chunk
            doc_id: Document ID (e.g., "2401.12345")
            title: Document title
            chunk_size: Chunk size (default: self.chunk_size)
            chunk_overlap: Overlap between chunks (default: self.chunk_overlap)

        Returns:
            List of chunk dictionaries with metadata
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        if self.use_semantic_chunking and self.semantic_chunker:
            chunks = self._semantic_chunk(text, doc_id, title)
        else:
            chunks = self._simple_chunk(text, doc_id, title, chunk_size, chunk_overlap)

        logger.info(f"Chunked text into {len(chunks)} chunks")
        return chunks

    def _simple_chunk(
        self,
        text: str,
        doc_id: str,
        title: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Simple character-based chunking.

        Args:
            text: Text to chunk
            doc_id: Document ID
            title: Document title
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size

            # Extract chunk
            chunk_text = text[start:end]

            # Add chunk with metadata
            chunks.append({
                "id": f"{doc_id}_{chunk_index}",
                "doc_id": doc_id,
                "title": title,
                "contents": chunk_text,
                "chunk_index": chunk_index
            })

            start = end - chunk_overlap
            chunk_index += 1

        return chunks

    def _semantic_chunk(
        self,
        text: str,
        doc_id: str,
        title: str
    ) -> List[Dict[str, Any]]:
        """
        Semantic chunking using chonkie.

        Args:
            text: Text to chunk
            doc_id: Document ID
            title: Document title

        Returns:
            List of chunk dictionaries
        """
        try:
            # Use chonkie's semantic chunker
            chunks = self.semantic_chunker.chunk(text)

            result = []
            for chunk_index, chunk in enumerate(chunks):
                result.append({
                    "id": f"{doc_id}_{chunk_index}",
                    "doc_id": doc_id,
                    "title": title,
                    "contents": chunk.text,
                    "chunk_index": chunk_index
                })

            return result

        except Exception as e:
            logger.error(f"Semantic chunking failed, falling back to simple: {e}")
            return self._simple_chunk(text, doc_id, title, self.chunk_size, self.chunk_overlap)

    def parse_and_chunk_pdf(
        self,
        pdf_path: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse PDF and chunk it in one step.

        Args:
            pdf_path: Path to PDF file
            chunk_size: Chunk size (default: self.chunk_size)
            chunk_overlap: Overlap between chunks (default: self.chunk_overlap)

        Returns:
            List of chunk dictionaries
        """
        doc = self.parse_pdf(pdf_path)
        chunks = self.chunk_text(
            doc['contents'],
            doc['id'],
            doc['title'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return chunks
