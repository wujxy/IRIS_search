#!/usr/bin/env python3
"""
IRIS Query Interface
Interactive script to query the IRIS knowledge base.

Usage:
    python scripts/iris_query.py "What machine learning methods are used?"
    python scripts/iris_query.py --index path/to/index.index --chunks path/to/chunks.jsonl "question"
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


logger = logging.getLogger(__name__)


def find_index_files(database_root: str, update_folder: str = None):
    """
    Find index files from the IRIS database.

    Args:
        database_root: Path to the IRIS database root
        update_folder: Specific update folder to use (optional)

    Returns:
        Tuple of (chunks_path, index_path)
    """
    database_root = Path(database_root)

    if update_folder:
        folder_path = database_root / update_folder
    else:
        folder_path = get_latest_update_folder(database_root)

    if folder_path is None:
        logger.error("No update folders found in the database")
        logger.info(f"Database root: {database_root}")
        return None, None

    logger.info(f"Using update folder: {folder_path}")

    # Try to find index_storage in update folder
    index_storage = folder_path / "index_storage"
    chunks_file = index_storage / "chunks.jsonl"
    index_file = index_storage / "index.index"

    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return None, None

    if not index_file.exists():
        logger.error(f"Index file not found: {index_file}")
        return None, None

    logger.info(f"Chunks file: {chunks_file}")
    logger.info(f"Index file: {index_file}")

    return chunks_file, index_file


def interactive_mode(qa_service: QAService, chunks_path: Path, index_path: Path):
    """
    Run interactive query mode.

    Args:
        qa_service: QA service instance
        chunks_path: Path to chunks file
        index_path: Path to index file
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
                index_path=index_path,
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
    index_path: Path,
    question: str
):
    """
    Run a single query and display the result.

    Args:
        qa_service: QA service instance
        chunks_path: Path to chunks file
        index_path: Path to index file
        question: Question to ask
    """
    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Query the knowledge base
    answer = qa_service.query_knowledge_base(
        question=question,
        index_path=index_path,
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
  python iris_query.py --chunks ./chunks.jsonl --index ./index.index "question"

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
        "--index",
        type=str,
        default=None,
        help="Path to index.index file (overrides auto-detection)"
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

    # Find index files
    if args.chunks and args.index:
        # Use explicit paths
        chunks_path = Path(args.chunks)
        index_path = Path(args.index)
    else:
        # Auto-detect from database
        chunks_path, index_path = find_index_files(database_root, args.update)

    if chunks_path is None or index_path is None:
        sys.exit(1)

    # Initialize QA service
    qa_service = QAService(
        ultrarag_path=config["ultrarag"]["ultrarag_path"],
        embedding_model=config["models"]["embedding_model_path"],
        reranker_model=config["models"]["reranker_model_path"],
        generation_model=config["models"]["llm_model_path"],
        vllm_base_url=config["models"]["vllm"]["base_url"],
        served_model_name=config["models"]["vllm"]["served_model_name"],
        system_prompt=config["qa"]["system_prompt"],
        index_backend=config["ultrarag"]["index_backend"]
    )

    # Check vLLM service
    if not qa_service.check_vllm_service(timeout=60):
        logger.error("vLLM service is not available")
        print("\nError: vLLM service is not running.")
        print("Please start the vLLM service first:")
        print(f"  cd {config['ultrarag']['ultrarag_path']}")
        print("  .venv/bin/python -m vllm.entrypoints.openai.api_server \\")
        print(f"    --model {config['models']['llm_model_path']} \\")
        print("    --host 127.0.0.1 --port 65504")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        interactive_mode(qa_service, chunks_path, index_path)

    # Single query mode
    elif args.question:
        single_query_mode(qa_service, chunks_path, index_path, args.question)

    else:
        # No question and not interactive - show help
        print("Error: Please provide a question or use --interactive mode.")
        print("\nUse --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
