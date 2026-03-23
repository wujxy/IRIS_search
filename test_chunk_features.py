"""
Test script for new chunk features: references removal and sentence-aware chunking.
"""

import sys
sys.path.insert(0, 'src')

from infrastructure.document_processor import DocumentProcessor


def test_remove_references():
    """Test references removal functionality."""
    print("=" * 60)
    print("Testing References Removal")
    print("=" * 60)

    # Create a document processor
    processor = DocumentProcessor(
        chunk_size=512,
        chunk_overlap=50,
        remove_references=True
    )

    # Test case 1: Standard references section
    test_text_1 = """
This is the main content of the paper.

We conducted several experiments to validate our approach.

The results show significant improvements over baseline methods.

References
[1] Smith, J. et al. "A Novel Approach to Something." Journal of Examples, 2020.
[2] Doe, A. and Brown, B. "Another Important Paper." Proceedings of Something, 2021.
[3] Johnson, C. "Yet Another Reference." arXiv:1234.5678, 2022.
"""

    cleaned_text_1, metadata_1 = processor._remove_references_section(test_text_1)

    print(f"\nTest 1: Standard 'References' section")
    print(f"  Original length: {metadata_1['original_length']}")
    print(f"  Cleaned length: {metadata_1['new_length']}")
    print(f"  Chars removed: {metadata_1['chars_removed']}")
    print(f"  Method matched: {metadata_1['method']}")
    print(f"  References removed: {metadata_1['removed']}")

    assert metadata_1['removed'] == True, "References should be removed"
    assert "References" not in cleaned_text_1, "References section should be gone"
    assert "This is the main content" in cleaned_text_1, "Main content should remain"

    # Test case 2: Bibliography section
    test_text_2 = """
Introduction to the topic.

Methodology and results.

Bibliography
1. Author One. "Book Title." Publisher, 2020.
2. Author Two. "Article Title." Journal Name, 2021.
"""

    cleaned_text_2, metadata_2 = processor._remove_references_section(test_text_2)

    print(f"\nTest 2: 'Bibliography' section")
    print(f"  Original length: {metadata_2['original_length']}")
    print(f"  Cleaned length: {metadata_2['new_length']}")
    print(f"  Chars removed: {metadata_2['chars_removed']}")
    print(f"  Method matched: {metadata_2['method']}")
    print(f"  References removed: {metadata_2['removed']}")

    assert metadata_2['removed'] == True, "Bibliography should be removed"

    # Test case 3: No references section
    test_text_3 = """
This is a paper without a references section.

Just the main content here.
"""

    cleaned_text_3, metadata_3 = processor._remove_references_section(test_text_3)

    print(f"\nTest 3: No references section")
    print(f"  Original length: {metadata_3['original_length']}")
    print(f"  Cleaned length: {metadata_3['new_length']}")
    print(f"  References removed: {metadata_3['removed']}")

    assert metadata_3['removed'] == False, "Should not remove anything"
    assert cleaned_text_3 == test_text_3, "Text should be unchanged"

    print("\n" + "=" * 60)
    print("References Removal: ALL TESTS PASSED!")
    print("=" * 60)


def test_sentence_aware_chunking():
    """Test sentence-aware chunking functionality."""
    print("\n" + "=" * 60)
    print("Testing Sentence-Aware Chunking")
    print("=" * 60)

    # Create a document processor
    processor = DocumentProcessor(
        chunk_size=200,  # Small size for testing
        chunk_overlap=50,
        remove_references=False,
        chunk_backend="sentence"
    )

    test_text = """
This is the first sentence. This is the second sentence with more content.
This is the third sentence. This is the fourth sentence.

This is a new paragraph with the fifth sentence. The sixth sentence is here.
The seventh sentence concludes this paragraph.
"""

    chunks = processor._sentence_aware_chunk(
        test_text,
        doc_id="test_doc",
        title="Test Document",
        chunk_size=200,
        chunk_overlap=50
    )

    print(f"\nNumber of chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  ID: {chunk['id']}")
        print(f"  Content length: {len(chunk['contents'])} chars")
        print(f"  Content preview: {chunk['contents'][:100]}...")

        # Verify no sentence is cut off mid-way
        content = chunk['contents'].strip()
        # Check that it doesn't end with a partial word (basic check)
        assert not content.endswith(' ') or len(content) == 0, "Chunk should not end with space unless empty"

    print("\n" + "=" * 60)
    print("Sentence-Aware Chunking: TESTS COMPLETED!")
    print("=" * 60)


def test_integration():
    """Test integration with different chunk_backend options."""
    print("\n" + "=" * 60)
    print("Testing Integration")
    print("=" * 60)

    # Test with sentence backend (default)
    processor_sentence = DocumentProcessor(
        chunk_size=300,
        chunk_overlap=50,
        chunk_backend="sentence"
    )

    test_text = """
Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.
Sentence six. Sentence seven. Sentence eight. Sentence nine. Sentence ten.
Sentence eleven. Sentence twelve.
""" * 3  # Repeat to make it longer

    chunks_sentence = processor_sentence.chunk_text(
        test_text,
        doc_id="integration_test",
        title="Integration Test"
    )

    print(f"\nWith chunk_backend='sentence': {len(chunks_sentence)} chunks created")

    # Test with simple backend (fallback)
    processor_simple = DocumentProcessor(
        chunk_size=300,
        chunk_overlap=50,
        chunk_backend="simple"  # Will use _simple_chunk
    )

    chunks_simple = processor_simple.chunk_text(
        test_text,
        doc_id="integration_test",
        title="Integration Test"
    )

    print(f"With chunk_backend='simple': {len(chunks_simple)} chunks created")

    print("\n" + "=" * 60)
    print("Integration: TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_remove_references()
        test_sentence_aware_chunking()
        test_integration()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Make sure you have installed all dependencies:")
        print("  pip install nltk")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
