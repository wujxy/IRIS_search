#!/usr/bin/env python3
"""
Test script to debug PDF processing issue with detailed error tracking
"""

import logging
from pathlib import Path
import sys
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import fitz  # pymupdf

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_simple_pdf_read():
    """Test simple PDF reading with PyMuPDF"""

    pdf_path = Path("/home/NagaiYoru/research/IRIS_papers/update_2026_03_19_1607/pdfs/2603.17964v1.pdf")

    print(f"\n{'='*60}")
    print(f"TEST 1: Simple PyMuPDF Read")
    print(f"{'='*60}\n")

    doc = None
    try:
        # Open PDF
        doc = fitz.open(str(pdf_path))
        print(f"✓ Opened PDF successfully")
        print(f"  - Number of pages: {len(doc)}")
        print(f"  - Metadata: {doc.metadata}")

        # Get first page
        first_page = doc[0]
        print(f"\n✓ Got first page")
        page_text = first_page.get_text()
        print(f"  - First page text length: {len(page_text)} chars")
        print(f"  - First 200 chars: {page_text[:200]}")

        # Try get_text("blocks")
        print(f"\n✓ Trying get_text('blocks')...")
        blocks = first_page.get_text("blocks")
        print(f"  - Number of blocks: {len(blocks)}")
        if blocks:
            print(f"  - First block preview: {blocks[0][4][:100] if len(blocks[0]) > 4 else 'N/A'}")

        # Close document
        doc.close()
        doc = None
        print(f"\n✓ Closed document")

        # Try to access after close (this should fail)
        try:
            _ = len(doc)
            print(f"✗ ERROR: Should have failed but didn't!")
        except Exception as e:
            print(f"✓ Expected error after close: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        if doc:
            try:
                doc.close()
            except:
                pass

def test_get_blocks_loop():
    """Test the problematic loop that uses get_text('blocks')"""

    pdf_path = Path("/home/NagaiYoru/research/IRIS_papers/update_2026_03_19_1607/pdfs/2603.17964v1.pdf")

    print(f"\n{'='*60}")
    print(f"TEST 2: get_text('blocks') Loop (Original Code Logic)")
    print(f"{'='*60}\n")

    doc = fitz.open(str(pdf_path))
    print(f"✓ Opened PDF: {len(doc)} pages")

    texts = []
    try:
        for pg in doc:
            try:
                print(f"  Processing page {pg}...")
                blocks = pg.get_text("blocks")
                print(f"    - Got {len(blocks)} blocks")
                if blocks:
                    blocks.sort(key=lambda b: (b[1], b[0]))
                    page_text = "\n".join([b[4] for b in blocks if b[4] and b[4].strip()])
                    if page_text.strip():
                        texts.append(page_text)
                        print(f"    - Added text: {len(page_text)} chars")
                else:
                    page_text = pg.get_text()
                    if page_text and page_text.strip():
                        texts.append(page_text)
                        print(f"    - Fallback: added {len(page_text)} chars")
            except Exception as e:
                print(f"    ✗ ERROR on page: {e}")
                traceback.print_exc()

        contents = "\n\n".join(texts)
        print(f"\n✓ Total contents: {len(contents)} chars")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
    finally:
        try:
            doc.close()
            print(f"\n✓ Closed document")
        except Exception as e:
            print(f"✗ Error closing: {e}")

if __name__ == "__main__":
    test_simple_pdf_read()
    test_get_blocks_loop()
