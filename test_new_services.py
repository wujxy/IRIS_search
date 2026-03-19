#!/usr/bin/env python3
"""
Test script for new IRIS services (independent of UltraRAG).
Tests Milvus, Embedding, Document Processor, and QA services.
"""

import asyncio
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src"))

from infrastructure.milvus_service import MilvusService
from infrastructure.embedding_service import EmbeddingService
from infrastructure.reranker_service import RerankerService
from infrastructure.document_processor import DocumentProcessor
from core.index_service import IndexService
from core.qa_service import QAService
from core.retriever import Retriever


def test_document_processor():
    """Test document processor (PDF parsing and chunking)."""
    print("=" * 60)
    print("TEST 1: Document Processor")
    print("=" * 60)

    processor = DocumentProcessor(
        chunk_size=256,
        chunk_overlap=50,
        use_title=True,
    )

    # Test with a sample text
    sample_text = """
    This is a sample document. It contains multiple paragraphs.
    The goal is to test the chunking functionality.

    This is the second paragraph. It adds more content to test.
    The chunker should split this text appropriately.
    """

    chunks = processor.chunk_text(
        text=sample_text,
        doc_id="test_doc",
        title="Sample Document",
    )

    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Content length: {len(chunk['contents'])} chars")
        print(f"  Preview: {chunk['contents'][:100]}...")

    print("\nDocument Processor Test: PASSED\n")
    return True


def test_milvus_service():
    """Test Milvus service (connection and basic operations)."""
    print("=" * 60)
    print("TEST 2: Milvus Service")
    print("=" * 60)

    try:
        milvus = MilvusService(
            uri="http://localhost:29901",
            collection_name="test_collection",
            embedding_dim=768,
        )

        # Test creating collection
        print("Creating test collection...")
        milvus.create_collection(dim=768, overwrite=True)

        # Test inserting data
        import numpy as np
        test_embeddings = np.random.rand(5, 768).astype(np.float32)
        test_chunks = [
            {"id": f"chunk_{i}", "contents": f"Test content {i}"}
            for i in range(5)
        ]

        print("Inserting test data...")
        milvus.insert(test_embeddings, test_chunks)

        # Test searching
        query_embedding = test_embeddings[0]
        results = milvus.search(query_embedding, top_k=3)

        print(f"Retrieved {len(results)} results")
        for result in results[:2]:
            print(f"  - {result.get('contents', 'N/A')[:50]}...")

        # Clean up
        milvus.drop_collection()

        print("\nMilvus Service Test: PASSED\n")
        return True

    except Exception as e:
        print(f"\nMilvus Service Test: FAILED - {e}\n")
        return False


def test_embedding_service():
    """Test embedding service (vLLM API call)."""
    print("=" * 60)
    print("TEST 3: Embedding Service")
    print("=" * 60)

    try:
        embed_svc = EmbeddingService(
            base_url="http://127.0.0.1:65503/v1",
            model_name="qwen3-embedding-0.6b",
        )

        test_texts = [
            "This is a test sentence.",
            "Another test for embedding generation.",
        ]

        print("Generating embeddings...")
        # Use sync wrapper for simplicity in test
        embeddings = embed_svc.encode_sync(test_texts)

        print(f"Generated embeddings: shape={embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")

        embed_svc.close()

        print("\nEmbedding Service Test: PASSED\n")
        return True

    except Exception as e:
        print(f"\nEmbedding Service Test: FAILED - {e}\n")
        print("  Note: Make sure vLLM embedding server is running on port 65503")
        return False


def test_reranker():
    """Test reranker service (CrossEncoder)."""
    print("=" * 60)
    print("TEST 4: Reranker Service")
    print("=" * 60)

    try:
        reranker = RerankerService(
            model_path="/home/NagaiYoru/LLM_model/bge-reranker-v2-m3",
            device="cpu",  # Change to "cuda:0" for GPU
        )

        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "This is about computer vision.",
            "Natural language processing deals with text.",
        ]

        print("Reranking passages...")
        reranked = reranker.rerank(query, passages, top_k=3)

        print(f"Top 3 reranked results:")
        for idx, score in reranked:
            print(f"  {idx+1}. Score: {score:.4f}")
            print(f"     Passage: {passages[idx][:60]}...")

        reranker.close()

        print("\nReranker Service Test: PASSED\n")
        return True

    except Exception as e:
        print(f"\nReranker Service Test: FAILED - {e}\n")
        print("  Note: Make sure reranker model path is correct")
        return False


def test_retriever():
    """Test retriever (integrates embedding, Milvus, reranker)."""
    print("=" * 60)
    print("TEST 5: Retriever (Integrated)")
    print("=" * 60)

    try:
        # Create services directly
        milvus_service = MilvusService(
            uri="http://localhost:29901",
            collection_name="test_retriever",
            embedding_dim=768
        )

        embedding_service = EmbeddingService(
            base_url="http://127.0.0.1:65503/v1",
            model_name="qwen3-embedding-0.6b"
        )

        retriever = Retriever(
            embedding_service=embedding_service,
            milvus_service=milvus_service,
            reranker_service=None,  # Skip reranker for this test
            default_top_k=3
        )

        # Test retrieval
        async def test_retrieve():
            query = "What is artificial intelligence?"
            results = await retriever.retrieve(query, mode="global", top_k=3)

            print(f"Retrieved {len(results)} passages for: '{query}'")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.get('contents', 'N/A')[:50]}...")

        asyncio.run(test_retrieve())

        print("\nRetriever Test: PASSED\n")
        return True

    except Exception as e:
        print(f"\nRetriever Test: FAILED - {e}\n")
        print("  Note: Make sure vLLM embedding and Milvus are running")
        return False


async def test_qa_service():
    """Test QA service (RAG with retrieval and generation)."""
    print("=" * 60)
    print("TEST 6: QA Service (RAG)")
    print("=" * 60)

    try:
        # Create QA service directly
        milvus_service = MilvusService(
            uri="http://localhost:29901",
            collection_name="test_qa",
            embedding_dim=768
        )

        embedding_service = EmbeddingService(
            base_url="http://127.0.0.1:65503/v1",
            model_name="qwen3-embedding-0.6b"
        )

        retriever = Retriever(
            embedding_service=embedding_service,
            milvus_service=milvus_service,
            reranker_service=None,  # Skip reranker for this test
            default_top_k=3
        )

        qa_service = QAService(
            retriever=retriever,
            generation_base_url="http://127.0.0.1:65504/v1",
            generation_model="llama3-3b-instruct"
        )

        # Test single query
        question = "What is the capital of France?"
        print(f"Question: {question}\n")

        answer = await qa_service.query(
            question=question,
            mode="global",
            top_k=3
        )

        print(f"\nAnswer: {answer}")

        # Test batch query
        print("\n" + "-" * 40)
        print("Testing batch query...")
        questions = [
            "What is Python?",
            "What is machine learning?",
        ]

        results = await qa_service.query_batch(
            questions=questions,
            mode="global",
            top_k=2
        )

        print("\nBatch results:")
        for result in results:
            print(f"  Q: {result['question'][:30]}...")
            print(f"  A: {result['answer'][:80]}...")

        # Test multi-turn conversation
        print("\n" + "-" * 40)
        print("Testing multi-turn conversation...")
        session_id = qa_service.create_conversation()

        answer1 = await qa_service.query_with_conversation(
            session_id=session_id,
            question="What is the largest planet?",
            mode="global"
        )

        answer2 = await qa_service.query_with_conversation(
            session_id=session_id,
            question="And what about the second largest?",
            mode="global"
        )

        print(f"\nConversation: {session_id[:8]}...")
        print(f"  Q1: What is the largest planet?")
        print(f"  A1: {answer1[:80]}...")
        print(f"  Q2: And what about the second largest?")
        print(f"  A2: {answer2[:80]}...")

        print("\nQA Service Test: PASSED\n")
        return True

    except Exception as e:
        print(f"\nQA Service Test: FAILED - {e}\n")
        print("  Note: Make sure vLLM embedding, generation, and Milvus are running")
        import traceback
        traceback.print_exc()
        return False


def test_specific_mode():
    """Test specific mode (paper_id filtering)."""
    print("=" * 60)
    print("TEST 7: Specific Mode (Paper ID Filtering)")
    print("=" * 60)

    try:
        config = {
            "embedding_base_url": "http://127.0.0.1:65503/v1",
            "embedding_model_name": "qwen3-embedding-0.6b",
            "milvus_uri": "http://localhost:29901",
            "milvus_collection_name": "test_specific",
            "milvus_embedding_dim": 768,
            "reranker_model_path": None,
            "top_k": 3,
        }

        from core.qa_service import create_qa_service_from_config
        qa_service = create_qa_service_from_config(config)

        question = "What is this paper about?"
        paper_id = "test_paper"

        print(f"Question: {question}")
        print(f"Paper ID: {paper_id}\n")

        answer = await qa_service.query(
            question=question,
            mode="specific",
            paper_id=paper_id,
            top_k=3
        )

        print(f"\nAnswer: {answer}")
        print("\nNote: Specific mode uses Milvus filter to restrict search")
        print("      to only chunks with matching doc_id field")

        print("\nSpecific Mode Test: PASSED\n")
        return True

    except Exception as e:
        print(f"\nSpecific Mode Test: FAILED - {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("IRIS New Services Test Suite")
    print("=" * 60)
    print("\nTesting services independent of UltraRAG framework\n")

    results = {
        "Document Processor": test_document_processor(),
        "Milvus Service": test_milvus_service(),
        "Embedding Service": test_embedding_service(),
        "Reranker Service": test_reranker(),
        "Retriever": test_retriever(),
    }

    # Async tests
    import asyncio
    results["QA Service"] = asyncio.run(test_qa_service())
    results["Specific Mode"] = asyncio.run(test_specific_mode())

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name:25s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
