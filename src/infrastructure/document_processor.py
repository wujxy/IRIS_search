"""
Document Processor for IRIS
PDF parsing and text chunking functionality.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    from chonkie import SemanticChunker
except ImportError:
    SemanticChunker = None

try:
    import nltk
except ImportError:
    nltk = None

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
        semantic_model: Optional[str] = None,
        remove_references: bool = True,
        chunk_backend: str = "sentence"
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Maximum chunk size in tokens (default: 512)
            chunk_overlap: Overlap between chunks (default: 50)
            use_semantic_chunking: Use semantic chunking (default: False)
            semantic_model: Model for semantic chunking (optional)
            remove_references: Remove references section (default: True)
            chunk_backend: Chunking strategy: sentence, recursive, or semantic (default: sentence)
        """
        if fitz is None:
            raise ImportError(
                "pymupdf is not installed. Install it with `pip install pymupdf`"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking
        self.remove_references = remove_references
        self.chunk_backend = chunk_backend

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
            f"overlap={chunk_overlap}, semantic={use_semantic_chunking}, "
            f"remove_references={remove_references}, chunk_backend={chunk_backend}"
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
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = None
        try:
            # 1. 打开 PDF
            doc = fitz.open(str(pdf_path))

            # 2. 提取标题（从文件名或第一页）
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

            # 3. 提取所有文本（使用更稳健的方法）
            contents = ""

            # 方法 1：尝试使用 get_text("blocks") 类似 UltraRAG
            try:
                texts = []
                for pg in doc:
                    try:
                        blocks = pg.get_text("blocks")
                        if blocks:
                            # 按位置排序（y0, x0）
                            blocks.sort(key=lambda b: (b[1], b[0]))
                            # 只取文本内容（b[4]）
                            page_text = "\n".join([b[4] for b in blocks if b[4] and b[4].strip()])
                            if page_text.strip():
                                texts.append(page_text)
                        else:
                            # 回退到 get_text()
                            page_text = pg.get_text()
                            if page_text and page_text.strip():
                                texts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract from page {pg}: {e}")
                        # 回退
                        try:
                            page_text = pg.get_text()
                            if page_text and page_text.strip():
                                texts.append(page_text)
                        except Exception:
                            pass

                contents = "\n\n".join(texts)

            except Exception as e:
                # 方法 2：回退到简单 get_text()
                logger.warning(f"Using fallback text extraction for {pdf_path.name}: {e}")
                contents = ""
                try:
                    for page in doc:
                        page_text = page.get_text()
                        if page_text:
                            contents += page_text + "\n"
                except Exception as e2:
                    logger.error(f"Text extraction failed: {e2}")
                    contents = ""

            # 5. 提取 paper ID
            paper_id = pdf_path.stem.replace('_', '.')

            # 在关闭文档前获取页数
            page_count = len(doc) if doc else 0

            logger.debug(f"Parsed PDF: {pdf_path.name}, pages={page_count}, chars={len(contents)}")

            # Remove references section if enabled
            if self.remove_references:
                contents, ref_metadata = self._remove_references_section(contents)
                if ref_metadata['removed']:
                    logger.info(f"References removed from {pdf_path.name}: "
                                f"{ref_metadata['chars_removed']} chars, "
                                f"method={ref_metadata['method']}")

            # 4. 关闭文档
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

            return {
                "id": paper_id,
                "title": title,
                "contents": contents.strip()
            }

        except FileNotFoundError:
            raise
        except Exception as e:
            # 改进：返回空内容而不是抛出异常，让流程可以继续
            logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            return {
                "id": pdf_path.stem,
                "title": pdf_path.stem,
                "contents": ""
            }

    def _remove_references_section(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Remove references/bibliography section from academic paper text.

        Uses multiple regex patterns to detect reference sections with high accuracy.

        Args:
            text: Full text content from PDF

        Returns:
            Tuple of (cleaned_text, metadata) where metadata contains:
            - removed: True if references were found and removed
            - method: The detection pattern that matched
            - position: Character position where cut occurred
            - original_length: Length before removal
            - new_length: Length after removal
        """
        original_length = len(text)

        # Define reference section patterns in priority order
        reference_patterns = [
            # Exact matches (highest priority)
            (r'^References\s*$', 'References'),
            (r'^Bibliography\s*$', 'Bibliography'),
            (r'^REFERENCES\s*$', 'REFERENCES'),
            (r'^BIBLIOGRAPHY\s*$', 'BIBLIOGRAPHY'),
            # Multi-language
            (r'^参考文献\s*$', '参考文献'),
            # Numbered variants
            (r'^\d+\.\s*References\s*$', 'Numbered References'),
            (r'^[IVX]+\.\s*References\s*$', 'Roman References'),
        ]

        # Try each pattern
        for pattern, method in reference_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.UNICODE)
            if match:
                cut_position = match.start()
                # Verify this looks like a reference section
                # by checking if there's significant text after it
                after_match = text[cut_position + len(match.group()):cut_position + 500]

                # Context validation: check for reference-like patterns
                reference_indicators = [
                    r'\[\d+\]',           # [1], [23], etc.
                    r'\(\d{4}\)',          # (2020), (1999), etc.
                    r'^\d+\.',            # 1. Author, Title (numbered refs)
                    r'^[A-Z][a-z]+,',     # Author, (et al)
                ]

                is_reference_section = any(
                    re.search(pattern, after_match, re.MULTILINE)
                    for pattern in reference_indicators
                )

                if is_reference_section or len(after_match.strip()) > 200:
                    cleaned_text = text[:cut_position].strip()
                    metadata = {
                        'removed': True,
                        'method': method,
                        'position': cut_position,
                        'original_length': original_length,
                        'new_length': len(cleaned_text),
                        'chars_removed': original_length - len(cleaned_text)
                    }
                    logger.info(f"Removed '{method}' section at position {cut_position}, "
                                f"{metadata['chars_removed']} chars removed")
                    return cleaned_text, metadata

        # No reference section found
        return text, {
            'removed': False,
            'method': None,
            'position': None,
            'original_length': original_length,
            'new_length': original_length,
            'chars_removed': 0
        }

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

        # Select chunking strategy based on chunk_backend
        if self.chunk_backend == "sentence":
            chunks = self._sentence_aware_chunk(text, doc_id, title, chunk_size, chunk_overlap)
        elif self.chunk_backend == "recursive":
            # Recursive chunking not yet implemented, fall back to sentence
            logger.warning("Recursive chunking not yet implemented, using sentence-aware chunking")
            chunks = self._sentence_aware_chunk(text, doc_id, title, chunk_size, chunk_overlap)
        elif self.use_semantic_chunking and self.semantic_chunker:
            chunks = self._semantic_chunk(text, doc_id, title)
        else:
            chunks = self._simple_chunk(text, doc_id, title, chunk_size, chunk_overlap)

        logger.info(f"Chunked text into {len(chunks)} chunks using {self.chunk_backend} backend")
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

    def _sentence_aware_chunk(
        self,
        text: str,
        doc_id: str,
        title: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Sentence-aware chunking that respects sentence boundaries.

        Splits text into sentences using NLTK, then groups sentences into chunks
        while respecting the chunk_size limit. Ensures no sentence is split
        across chunks and overlap includes complete sentences.

        Args:
            text: Text to chunk
            doc_id: Document ID
            title: Document title
            chunk_size: Maximum chunk size (in characters)
            chunk_overlap: Desired overlap between chunks (in characters)

        Returns:
            List of chunk dictionaries with metadata
        """
        if nltk is None:
            logger.warning("NLTK not installed, falling back to simple chunking")
            return self._simple_chunk(text, doc_id, title, chunk_size, chunk_overlap)

        # Download NLTK punkt data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK data: {e}, falling back to simple chunking")
                return self._simple_chunk(text, doc_id, title, chunk_size, chunk_overlap)

        # Split text into sentences
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}, falling back to simple chunking")
            return self._simple_chunk(text, doc_id, title, chunk_size, chunk_overlap)

        if not sentences:
            return []

        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        # Keep track of sentences for overlap (store sentence indices)
        overlap_sentences = []  # List of (sentence, length) tuples

        for sentence in sentences:
            sentence_len = len(sentence)

            # Check if adding this sentence would exceed chunk size
            if current_chunk and current_length + sentence_len > chunk_size:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)

                chunks.append({
                    "id": f"{doc_id}_{chunk_index}",
                    "doc_id": doc_id,
                    "title": title,
                    "contents": chunk_text,
                    "chunk_index": chunk_index
                })

                chunk_index += 1

                # Handle overlap: keep some sentences from the end of current chunk
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break

                current_chunk = overlap_sentences.copy()
                current_length = overlap_length
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_len

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "id": f"{doc_id}_{chunk_index}",
                "doc_id": doc_id,
                "title": title,
                "contents": chunk_text,
                "chunk_index": chunk_index
            })

        logger.debug(f"Sentence-aware chunking created {len(chunks)} chunks from {len(sentences)} sentences")
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
