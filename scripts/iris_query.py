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
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add src to path for new structure
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from utils.helpers import load_config, setup_logging
from services.deploy_service import DeployService
from services.paper_service import PaperService

# New independent services
from infrastructure.milvus_service import MilvusService
from infrastructure.embedding_service import EmbeddingService
from core.retriever import Retriever
from core.qa_service import QAService


logger = logging.getLogger(__name__)


def find_chunks_file(database_root: str, update_folder: str = None, use_master: bool = True):
    """
    Find chunks file from the IRIS database.

    NOTE: chunks.jsonl is deprecated and not used for retrieval.
    Milvus is used for all retrieval operations. This function
    returns None as the chunks file is no longer needed.

    Args:
        database_root: Path to IRIS database root (unused)
        update_folder: Specific update folder (unused)
        use_master: Use master collection (unused)

    Returns:
        None (chunks file is not used)
    """
    # The current IRIS implementation uses Milvus for all retrieval
    # chunks.jsonl files are no longer needed for querying
    logger.debug("Chunks file lookup skipped (using Milvus for retrieval)")
    return None


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


async def single_query_mode(
    qa_service: QAService,
    paper_service: PaperService,
    question: str,
    mode: str = "global",
    paper_id: str = None
):
    """
    Run a single query with specified mode.

    Args:
        qa_service: QA service instance
        paper_service: Paper service instance
        question: Question to ask
        mode: Query mode (global/specific)
        paper_id: Paper ID for specific mode
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

    # Query knowledge base (new async interface)
    answer = await qa_service.query(
        question=question,
        chunks_path=None,  # No longer used
        mode=mode,
        paper_id=paper_id,
        top_k=5
    )

    if answer:
        print(f"\nAnswer:\n{answer}\n")
    else:
        print("\nNo answer found. Please try rephrasing your question.\n")


async def interactive_mode(
    qa_service: QAService,
    paper_service: PaperService,
    default_mode: str = "global"
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
                answer = await qa_service.query(
                    question=question,
                    chunks_path=None,
                    mode="specific",
                    paper_id=current_paper_id,
                    top_k=5
                )
            else:
                answer = await qa_service.query(
                    question=question,
                    chunks_path=None,
                    mode="global",
                    paper_id=None,
                    top_k=5
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


async def main():
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
        default=None,
        help="Milvus collection name (default: uses master_collection from config)"
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

    # Get collection name - use collection_name from config (same as run_update_cycle)
    collection_name = args.collection or config["milvus"]["collection_name"]
    logger.info(f"Using collection: {collection_name}")

    # Initialize new independent services
    milvus_service = MilvusService(
        uri=config["milvus"]["uri"],
        collection_name=collection_name,
        embedding_dim=config["milvus"].get("embedding_dim", 1024)
    )

    embedding_service = EmbeddingService(
        base_url=config["embedding"]["base_url"],
        model_name=config["embedding"]["model_name"],
        batch_size=config["embedding"].get("batch_size", 32)
    )

    # Optional reranker
    reranker_service = None
    if config.get("reranker", {}).get("enabled", False):
        reranker_service = RerankerService(
            model_path=config["reranker"]["model_path"],
            batch_size=config["reranker"].get("batch_size", 16),
            device=config["reranker"].get("device", "cpu")
        )

    # Create retriever
    retriever = Retriever(
        embedding_service=embedding_service,
        milvus_service=milvus_service,
        reranker_service=reranker_service,
        default_top_k=config["qa"].get("top_k", 5)
    )

    # Initialize QA service
    qa_service = QAService(
        retriever=retriever,
        base_url=config["qa"]["base_url"],
        model_name=config["qa"]["model_name"],
        system_prompt=config["qa"]["system_prompt"],
        temperature=config["qa"].get("temperature", 0.7),
        max_tokens=config["qa"].get("max_tokens", 2048)
    )

    # Infrastructure management
    deploy_server = DeployService(config)
    infrastructure_auto_started = False

    # Check infrastructure status
    milvus_running = deploy_server.milvus_control.search()
    index_vllm_running = deploy_server.index_vllm.search(timeout=5)
    qa_vllm_running = deploy_server.qa_vllm.search(timeout=5)

    logger.info(f"Infrastructure status: Milvus={milvus_running}, Index vLLM={index_vllm_running}, QA vLLM={qa_vllm_running}")

    # Start infrastructure if needed or requested
    if args.start_infra or not (milvus_running and index_vllm_running and qa_vllm_running):
        if not (milvus_running and index_vllm_running and qa_vllm_running):
            logger.info("Infrastructure not fully running, starting...")
        else:
            logger.info("Starting infrastructure as requested...")

        if not deploy_server.start_infrastructure():
            logger.error("Failed to start infrastructure")
            sys.exit(1)
        infrastructure_auto_started = True

    try:
        # Run query
        if args.interactive:
            await interactive_mode(
                qa_service=qa_service,
                paper_service=paper_service,
                default_mode=args.mode
            )
        elif args.question:
            await single_query_mode(
                qa_service=qa_service,
                paper_service=paper_service,
                question=args.question,
                mode=args.mode,
                paper_id=args.paper_id
            )
        else:
            print("Error: Please provide a question or use --interactive mode.")
            return
    finally:
        # Stop infrastructure if auto-started
        if infrastructure_auto_started:
            logger.info("Stopping auto-started infrastructure...")
            deploy_server.stop_infrastructure()


if __name__ == "__main__":
    asyncio.run(main())
