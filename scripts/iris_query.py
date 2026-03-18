#!/usr/bin/env python3
"""
IRIS Query Interface
Interactive script to query the IRIS knowledge base.

Usage:
    python scripts/iris_query.py "What machine learning methods are used?"
    python scripts/iris_query.py --list-papers
    python scripts/iris_query.py --mode specific --paper-id 2403.15570 "What is the main finding?"
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
from services.paper_service import PaperService


logger = logging.getLogger(__name__)


def find_chunks_file(database_root: str, update_folder: str = None):
    """
    Find chunks file from the IRIS database.

    Args:
        database_root: Path to IRIS database root
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
        logger.error("No update folders found in database")
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


def list_papers_mode(paper_service: PaperService, limit: int = 20, category: str = None):
    """
    List papers in database with time sorting.

    Args:
        paper_service: Paper service instance
        limit: Max number of papers to display
        category: Filter by category (optional)
    """
    print("\n" + "=" * 80)
    print("IRIS Literature Database")
    print("=" * 80)

    # Query papers
    if category:
        papers = paper_service.search_papers(category=category, limit=limit)
        print(f"\nCategory: {category}")
    else:
        papers = paper_service.list_papers(limit=limit, order_by="published", reverse=True)
        print(f"\nShowing {len(papers)} papers (sorted by publication date)")

    if not papers:
        print("\nNo papers found.")
        return

    # Display papers
    print(f"\n{'ID':<12} {'Date':<12} {'Category':<15} {'Title':<30}")
    print("-" * 80)

    for paper in papers:
        paper_id = paper.get('paper_id', 'N/A')
        published = paper.get('published', 'N/A')[:10] if paper.get('published') else 'N/A'
        cat = paper.get('primary_category', 'N/A')
        title = paper.get('title', 'N/A')
        title = title[:27] + '...' if len(title) > 30 else title

        print(f"{paper_id:<12} {published:<12} {cat:<15} {title}")

    print(f"\nTotal: {len(papers)} papers")
    print("Use --paper-id <id> for specific queries")
    print("Use --help for more options")


def single_query_mode(
    qa_service: QAService,
    paper_service: PaperService,
    chunks_path: Path,
    question: str,
    mode: str = "global",
    paper_id: str = None,
    collection_name: str = "iris_papers"
):
    """
    Run a single query with specified mode.

    Args:
        qa_service: QA service instance
        paper_service: Paper service instance
        chunks_path: Path to chunks file
        question: Question to ask
        mode: Query mode (global/specific)
        paper_id: Paper ID for specific mode
        collection_name: Milvus collection name
    """
    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Specific mode validation
    if mode == "specific":
        if not paper_id:
            print("\nError: --paper-id is required for specific mode")
            print("Use --list-papers to see available papers.")
            return

        # Verify paper exists
        paper = paper_service.get_paper_by_id(paper_id)
        if not paper:
            print(f"\nError: Paper {paper_id} not found in database")
            print("Use --list-papers to see available papers.")
            return

        print(f"Querying specific paper: {paper.get('title', 'N/A')}")

    # Query knowledge base
    if mode == "specific":
        answer = qa_service.query_knowledge_base_with_mode(
            question=question,
            chunks_path=chunks_path,
            mode="specific",
            paper_id=paper_id,
            collection_name=collection_name
        )
    else:
        answer = qa_service.query_knowledge_base(
            question=question,
            chunks_path=chunks_path
        )

    if answer:
        print(f"\nAnswer:\n{answer}\n")
    else:
        print("\nNo answer found. Please try rephrasing your question.\n")


def interactive_mode(
    qa_service: QAService,
    paper_service: PaperService,
    chunks_path: Path,
    default_mode: str = "global",
    collection_name: str = "iris_papers"
):
    """
    Run interactive query mode with mode switching.

    Args:
        qa_service: QA service instance
        paper_service: Paper service instance
        chunks_path: Path to chunks file
        default_mode: Default query mode
        collection_name: Milvus collection name
    """
    print("\n" + "=" * 60)
    print("IRIS Interactive Query Mode")
    print("=" * 60)
    print(f"\nCurrent mode: {default_mode}")
    print("Commands:")
    print("  /mode <global|specific>  - Switch query mode")
    print("  /paper <id>             - Set paper for specific mode")
    print("  /list                    - List available papers")
    print("  /search <keyword>        - Search papers by keyword")
    print("  /quit                    - Exit")
    print("\nEnter your questions about literature.\n")

    current_mode = default_mode
    current_paper_id = None

    while True:
        try:
            prompt = f"IRIS[{current_mode}]"
            if current_mode == "specific" and current_paper_id:
                paper = paper_service.get_paper_by_id(current_paper_id)
                paper_title = paper.get('title', 'N/A')[:20] + '...' if paper else 'N/A'
                prompt += f" ({current_paper_id}: {paper_title})"

            question = input(f"{prompt}> ").strip()

            if not question:
                continue

            # Command handling
            if question.lower() in ['/quit', '/q', 'quit', 'q']:
                print("\nExiting IRIS query mode...")
                break

            elif question.lower() == '/list':
                list_papers_mode(paper_service, limit=10)
                continue

            elif question.lower().startswith('/mode '):
                new_mode = question.split()[1].lower()
                if new_mode in ['global', 'specific']:
                    current_mode = new_mode
                    if current_mode == 'global':
                        current_paper_id = None
                    print(f"\nMode switched to: {current_mode}")
                else:
                    print("\nUse /paper <id> to specify paper for specific mode")
                continue

            elif question.lower().startswith('/paper '):
                if current_mode != 'specific':
                    current_mode = 'specific'
                    print("\nMode switched to: specific")
                current_paper_id = question.split()[1]
                paper = paper_service.get_paper_by_id(current_paper_id)
                if paper:
                    print(f"\nPaper set: {paper.get('title', 'N/A')}")
                else:
                    print(f"\nError: Paper {current_paper_id} not found")
                    current_paper_id = None
                continue

            elif question.lower().startswith('/search '):
                keyword = question.split(maxsplit=1)[1] if len(question.split()) > 1 else ''
                if keyword:
                    papers = paper_service.search_papers(keyword=keyword, limit=10)
                    print(f"\nSearch results for '{keyword}':")
                    for paper in papers:
                        print(f"  {paper['paper_id']}: {paper['title']}")
                    print(f"\nFound {len(papers)} papers")
                continue

            # Regular query
            print(f"\nQuestion: {question}")
            print("-" * 40)

            if current_mode == "specific":
                if not current_paper_id:
                    print("\nError: Please set a paper using /paper <id>")
                    continue
                answer = qa_service.query_knowledge_base_with_mode(
                    question=question,
                    chunks_path=chunks_path,
                    mode="specific",
                    paper_id=current_paper_id,
                    collection_name=collection_name
                )
            else:
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IRIS Query Interface - Ask questions about literature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all papers
  python iris_query.py --list-papers

  # List papers by category
  python iris_query.py --list-papers --category astro-ph.HE --limit 10

  # Search papers
  python iris_query.py --search "neutrino"

  # Global mode query
  python iris_query.py --mode global "What machine learning methods are used?"

  # Specific mode query
  python iris_query.py --mode specific --paper-id 2403.15570 "What is the main finding?"

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
        action="store_true",
        help="List available update folders"
    )
    parser.add_argument(
        "--list-papers",
        action="store_true",
        help="List all papers in database with IDs and titles"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["global", "specific"],
        default="global",
        help="Query mode: global (all papers) or specific (single paper)"
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        default=None,
        help="Paper ID for specific mode (e.g., 2403.15570)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of papers to list (default: 20)"
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search papers by keyword"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (e.g., astro-ph.HE, hep-ph)"
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

    # Initialize PaperService
    paper_db_path = config["storage"].get("paper_db_path",
        str(Path(database_root) / "iris_papers.db"))
    paper_service = PaperService(paper_db_path)

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

    # List papers mode
    if args.list_papers:
        list_papers_mode(paper_service, limit=args.limit, category=args.category)
        return

    # Search papers mode
    if args.search:
        papers = paper_service.search_papers(keyword=args.search, limit=args.limit)
        print(f"\nSearch results for '{args.search}':")
        for paper in papers:
            print(f"  {paper['paper_id']}: {paper['title']}")
        print(f"\nFound {len(papers)} papers")
        return

    # Find chunks file
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
                interactive_mode(qa_service, paper_service, chunks_path,
                              default_mode=args.mode, collection_name=args.collection)
            elif args.question:
                single_query_mode(qa_service, paper_service, chunks_path, args.question,
                                  mode=args.mode, paper_id=args.paper_id,
                                  collection_name=args.collection)
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
        interactive_mode(qa_service, paper_service, chunks_path,
                      default_mode=args.mode, collection_name=args.collection)

    # Single query mode
    elif args.question:
        single_query_mode(qa_service, paper_service, chunks_path, args.question,
                          mode=args.mode, paper_id=args.paper_id,
                          collection_name=args.collection)

    else:
        # No question and not interactive - show help
        print("Error: Please provide a question, use --list-papers, or use --interactive mode.")
        print("\nUse --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
