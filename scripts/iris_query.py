#!/usr/bin/env python3
"""
IRIS Query Interface
Interactive script to query the IRIS knowledge base.

Usage:
    python scripts/iris_query.py "What machine learning methods are used?"
    python scripts/iris_query.py --chunks path/to/chunks.jsonl --collection iris_papers "question"
    python scripts/iris_query.py --interactive
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, setup_logging, get_latest_update_folder
from services.qa_service import QAService
from services.deploy_service import DeployService


logger = logging.getLogger(__name__)


def find_chunks_file(database_root: str, update_folder: str = None):
    """
    Find chunks file from the IRIS database.

    Args:
        database_root: Path to the IRIS database root
        update_folder: Specific update folder to use (optional)

    Returns:
        Path to chunks.jsonl file, or None if not found
    """
    database_root = Path(database_root)

    if update_folder:
        folder_path = database_root / update_folder
    else:
        folder_path = get_latest_update_folder(database_root)

    if folder_path is None:
        logger.error("No update folders found in the database")
        logger.info(f"Database root: {database_root}")
        return None

    logger.info(f"Using update folder: {folder_path}")

    # Try to find index_storage in update folder
    index_storage = folder_path / "index_storage"
    chunks_file = index_storage / "chunks.jsonl"

    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return None

    logger.info(f"Chunks file: {chunks_file}")

    return chunks_file


def interactive_mode(qa_service: QAService, chunks_path: Path):
    """
    Run interactive query mode.

    Args:
        qa_service: QA service instance
        chunks_path: Path to chunks file
    """
    print("\n" + "=" * 60)
    print("IRIS Interactive Query Mode")
    print("=" * 60)
    print("\nEnter your questions about the literature.")
    print("Type 'quit' or 'exit' to exit.\n")

    while True:
        try:
            question = input("IRIS> ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nExiting IRIS query mode...")
                break

            print(f"\nQuestion: {question}")
            print("-" * 40)

            # Query the knowledge base
            answer = qa_service.query_knowledge_base(
                question=question,
                chunks_path=chunks_path
            )

            if answer:
                print(f"\nAnswer: {answer}\n")
            else:
                print("\nNo answer found. Please try rephrasing your question.\n")

        except KeyboardInterrupt:
            print("\n\nExiting IRIS query mode...")
            break
        except Exception as e:
            logger.error(f"Error during query: {e}")
            print(f"\nError: {e}\n")


def single_query_mode(
    qa_service: QAService,
    chunks_path: Path,
    question: str,
    collection_name: str = "iris_papers"
):
    """
    Run a single query and display the result.

    Args:
        qa_service: QA service instance
        chunks_path: Path to chunks file
        question: Question to ask
        collection_name: Milvus collection name
    """
    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Query the knowledge base
    answer = qa_service.query_knowledge_base(
        question=question,
        chunks_path=chunks_path
    )

    if answer:
        print(f"\nAnswer:\n{answer}\n")
    else:
        print("\nNo answer found. Please try rephrasing your question.\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IRIS Query Interface - Ask questions about the literature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query (uses latest update)
  python iris_query.py "What machine learning methods are used?"

  # Single query with specific update folder
  python iris_query.py --update update_2026_03_15_1200 "What are the main findings?"

  # Single query with explicit paths
  python iris_query.py --chunks ./chunks.jsonl --collection iris_papers "question"

  # Interactive mode
  python iris_query.py --interactive

  # List available updates
  python iris_query.py --list-updates
        """
    )

    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="Question to ask (optional, use --interactive for interactive mode)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to IRIS database (overrides config)"
    )
    parser.add_argument(
        "--update",
        type=str,
        default=None,
        help="Specific update folder to use (e.g., update_2026_03_15_1200)"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="Path to chunks.jsonl file (overrides auto-detection)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="iris_papers",
        help="Milvus collection name"
    )
    parser.add_argument(
        "--start-infra",
        action="store_true",
        help="Start infrastructure (Milvus + vLLM) before query"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--list-updates",
        "-l",
        action="store_true",
        help="List available update folders"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override database path if specified
    database_root = args.database or config["storage"]["database_root"]

    # Setup logging
    setup_logging(
        log_dir=config["logging"]["log_dir"],
        level=args.log_level
    )

    # List updates mode
    if args.list_updates:
        print("\nAvailable Updates:\n")
        database_root = Path(database_root)
        update_folders = sorted(database_root.glob("update_*"), reverse=True)

        if not update_folders:
            print("  No updates found.")
        else:
            for folder in update_folders:
                metadata_file = folder / "metadata.json"
                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                            print(f"  {folder.name}")
                            print(f"    Time: {data.get('timestamp', 'N/A')}")
                            print(f"    New papers: {len(data.get('new_papers', []))}")
                    except:
                        print(f"  {folder.name} (metadata not readable)")
                else:
                    print(f"  {folder.name} (no metadata)")
        return

    # Find chunks file (Milvus: no index file needed)
    if args.chunks:
        chunks_path = Path(args.chunks)
    else:
        # Auto-detect from database
        chunks_path = find_chunks_file(database_root, args.update)

    if chunks_path is None:
        logger.error("Could not find chunks file")
        sys.exit(1)

    # Initialize QA service
    qa_service = QAService(
        ultrarag_path=config["ultrarag"]["ultrarag_path"],
        embedding_model=config["models"]["embedding_model_path"],
        reranker_model=config["models"]["reranker_model_path"],
        generation_model=config["models"]["llm_model_path"],
        vllm_base_url=config["models"]["vllm"]["base_url"],
        index_vllm_base_url=config["models"]["vllm"]["index"]["base_url"],
        served_model_name=config["models"]["vllm"]["served_model_name"],
        collection_name=args.collection
    )

    # Check vLLM service
    if args.start_infra:
        # 独立运行时启动基础设施（Milvus + vLLM 索引模型 + vLLM QA 模型）
        logger.info("Starting infrastructure...")
        deploy_server = DeployService(config)

        if not deploy_server.start_infrastructure():
            logger.error("Failed to start infrastructure")
            sys.exit(1)

        try:
            # Run query
            if args.interactive:
                interactive_mode(qa_service, chunks_path)
            elif args.question:
                single_query_mode(qa_service, chunks_path, args.question)
            else:
                print("Error: Please provide a question or use --interactive mode.")
                sys.exit(1)
        finally:
            # Stop infrastructure when done
            logger.info("Stopping infrastructure...")
            deploy_server.stop_infrastructure()
            return

    # If not starting infrastructure, check if services are available
    if not args.start_infra:
        logger.info("Not starting infrastructure - assuming services are already running")

    # Interactive mode
    if args.interactive:
        interactive_mode(qa_service, chunks_path)

    # Single query mode
    elif args.question:
        single_query_mode(qa_service, chunks_path, args.question)

    else:
        # No question and not interactive - show help
        print("Error: Please provide a question or use --interactive mode.")
        print("\nUse --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
