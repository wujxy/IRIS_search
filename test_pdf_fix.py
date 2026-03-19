#!/usr/bin/env python3
"""
Test script to verify PDF processing fix
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from infrastructure.document_processor import DocumentProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_fixed_pdf_processing():
    """Test fixed PDF processing"""

    pdf_path = Path("/home/NagaiYoru/research/IRIS_papers/update_2026_03_19_1607/pdfs/2603.17964v1.pdf")

    print(f"\n{'='*60}")
    print(f"Testing Fixed DocumentProcessor")
    print(f"{'='*60}\n")

    processor = DocumentProcessor(
        chunk_size=512,
        chunk_overlap=50,
        use_semantic_chunking=False
    )

    # Test parse_pdf
    print("[1] Testing parse_pdf()...")
    result = processor.parse_pdf(pdf_path)

    print(f"  - ID: {result['id']}")
    print(f"  - Title: {result['title'][:80]}...")
    print(f"  - Contents length: {len(result['contents'])} chars")

    if result['contents']:
        print(f"  - ✓ Contents extracted successfully!")
        print(f"  - Preview (first 200 chars):")
        print(f"    {result['contents'][:200]}")
    else:
        print(f"  - ✗ ERROR: Contents is empty!")

    # Test parse_and_chunk_pdf
    print(f"\n[2] Testing parse_and_chunk_pdf()...")
    chunks = processor.parse_and_chunk_pdf(pdf_path)

    print(f"  - Number of chunks: {len(chunks)}")

    if chunks:
        print(f"  - ✓ Chunks generated successfully!")
        print(f"  - First chunk length: {len(chunks[0]['contents'])} chars")
        print(f"  - First chunk preview (first 150 chars):")
        print(f"    {chunks[0]['contents'][:150]}")
    else:
        print(f"  - ✗ ERROR: No chunks generated!")

    print(f"\n{'='*60}")
    print(f"Test Complete")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    test_fixed_pdf_processing()
