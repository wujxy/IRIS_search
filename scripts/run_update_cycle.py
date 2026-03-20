#!/usr/bin/env python3
"""
IRIS Update Cycle Orchestrator
Main script to run a complete IRIS update cycle.

Usage:
    python scripts/run_update_cycle.py
    python scripts/run_update_cycle.py --config configs/config.yaml
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src"))

from infrastructure.milvus_service import MilvusService
from infrastructure.embedding_service import EmbeddingService
from infrastructure.reranker_service import RerankerService
from infrastructure.document_processor import DocumentProcessor
from core.retriever import create_retriever_from_config, Retriever
from core.index_service import create_index_service_from_config, IndexService
from core.qa_service import create_qa_service_from_config, QAService
from services.arxiv_service import ArxivService
from services.email_service import EmailService
from services.paper_service import PaperService
from services.deploy_service import DeployService
from utils.helpers import (
    load_config,
    load_questions,
    setup_logging,
    create_update_folder,
    save_metadata,
    save_update_log,
    save_summary_log,
    save_knowledge_log
)


logger = logging.getLogger(__name__)


def generate_summary_markdown(papers_with_answers: list) -> str:
    """
    Generate markdown summary from papers and their QA answers.

    Args:
        papers_with_answers: List of papers with QA results

    Returns:
        Markdown formatted summary string
    """
    markdown = "# Paper Summaries\n\n"
    markdown += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += "---\n\n"

    for paper_data in papers_with_answers:
        paper = paper_data["paper"]
        answers = paper_data.get("answers", {})

        markdown += f"## {paper['title']}\n\n"
        markdown += f"**Authors:** {', '.join(paper['authors'][:5])}\n\n"
        markdown += f"**arXiv ID:** {paper['entry_id']}\n\n"

        # Add summary from abstract
        markdown += "### Abstract Summary\n\n"
        markdown += f"{paper['summary'][:500]}...\n\n"

        # Add QA answers
        markdown += "### Key Information\n\n"
        for q_key, q_text in [
            ("q1", "Main Problem"),
            ("q2", "Key Contributions"),
            ("q3", "Methods"),
            ("q4", "Important Concepts"),
            ("q5", "Research Directions")
        ]:
            answer = answers.get(q_key, "N/A")
            markdown += f"**{q_text}:**\n{answer}\n\n"

        markdown += "---\n\n"

    return markdown


def generate_knowledge_markdown(papers_with_answers: list) -> str:
    """
    Generate knowledge extraction markdown from papers and their QA answers.

    Args:
        papers_with_answers: List of papers with QA results

    Returns:
        Markdown formatted knowledge string
    """
    markdown = "# Knowledge Extraction\n\n"
    markdown += f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += "---\n\n"

    for paper_data in papers_with_answers:
        paper = paper_data["paper"]
        answers = paper_data.get("answers", {})

        markdown += f"## {paper['title']}\n\n"

        # Extract knowledge from answers
        contributions = answers.get("q2", "")
        methods = answers.get("q3", "")
        concepts = answers.get("q4", "")
        directions = answers.get("q5", "")

        markdown += "### Key Contributions\n\n"
        markdown += f"{contributions}\n\n"

        markdown += "### Methods and Techniques\n\n"
        markdown += f"{methods}\n\n"

        markdown += "### Important Concepts\n\n"
        markdown += f"{concepts}\n\n"

        markdown += "### Future Research Directions\n\n"
        markdown += f"{directions}\n\n"

        markdown += "---\n\n"

    return markdown


def generate_update_log(metadata: dict, update_folder: Path, failed_downloads: list = None) -> str:
    """
    Generate update log markdown.

    Args:
        metadata: Metadata dictionary
        update_folder: Path to update folder
        failed_downloads: List of papers that failed to download (optional)

    Returns:
        Markdown formatted log string
    """
    markdown = "# IRIS Update Log\n\n"
    markdown += f"Timestamp: {metadata['timestamp']}\n\n"
    markdown += f"Update Folder: `{update_folder}`\n\n"
    markdown += "---\n\n"

    markdown += "## Summary\n\n"
    markdown += f"- Total papers found: {metadata['total_papers']}\n"
    markdown += f"- New papers: {len(metadata['new_papers'])}\n"
    markdown += f"- Duplicate papers: {len(metadata['duplicate_papers'])}\n"
    markdown += f"- Review papers (excluded): {len(metadata['review_papers'])}\n"
    if failed_downloads is not None:
        markdown += f"- Download failed (skipped): {len(failed_downloads)}\n\n"
    else:
        markdown += "\n"

    if metadata['new_papers']:
        markdown += "## New Papers\n\n"
        for paper in metadata['new_papers']:
            markdown += f"- **{paper['title']}**\n"
            markdown += f"  - Authors: {', '.join(paper['authors'][:3])}\n"
            markdown += f"  - arXiv: {paper['entry_id']}\n\n"

    if failed_downloads:
        markdown += "## Download Failed Papers\n\n"
        for paper in failed_downloads:
            markdown += f"- **{paper['title']}**\n"
            markdown += f"  - Authors: {', '.join(paper['authors'][:3])}\n"
            markdown += f"  - arXiv: {paper['entry_id']}\n"
            markdown += f"  - Reason: PDF download failed\n\n"

    return markdown


def run_update_cycle(config: dict):
    """
    Run a complete IRIS update cycle.

    Args:
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Starting IRIS Update Cycle")
    logger.info("=" * 60)

    try:
        # Step 1: Create update folder
        logger.info("\n[Step 1] Creating update folder...")
        update_folder = create_update_folder(config["storage"]["database_root"])
        logger.info(f"Update folder: {update_folder}")

        # Step 2: Initialize PaperService and load existing papers from SQLite
        logger.info("\n[Step 2] Initializing PaperService and loading existing papers...")
        paper_service = PaperService(config["storage"]["paper_db_path"])
        existing_ids = paper_service.get_all_entry_ids()
        logger.info(f"Found {len(existing_ids)} existing papers in SQLite database")

        # Step 3: Initialize arXiv service
        logger.info("\n[Step 3] Initializing arXiv service...")
        arxiv_config = config["arxiv"]
        arxiv_service = ArxivService(
            keywords=arxiv_config["keywords"],
            max_results=arxiv_config["max_results_per_keyword"],
            sort_by=arxiv_config["sort_by"],
            review_keywords=config["filtering"]["review_keywords"],
            exclude_reviews=config["filtering"]["exclude_reviews"],
            database_root=config["storage"]["database_root"]
        )

        # Step 4: Manage papers (search, filter, download, export via arxiv_service)
        logger.info("\n[Step 4] Managing papers via arxiv_service...")
        pdf_dir = update_folder / "pdfs"
        management_result = arxiv_service.manage_papers(
            save_dir=pdf_dir,
            target_count=arxiv_config["max_results_per_keyword"],
            existing_ids=existing_ids  # Pass SQLite-derived IDs
        )

        all_papers = management_result["papers"]
        downloaded_papers = management_result["downloaded"]
        new_papers = [p for p in all_papers if p.get("status") == "new"]

        # Step 5: Save metadata
        logger.info("\n[Step 5] Saving metadata...")
        save_metadata(all_papers, update_folder / "metadata.json")

        # Step 6: Generate update log
        logger.info("\n[Step 6] Generating update log...")
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_papers": len(all_papers),
            "new_papers": [p for p in all_papers if p.get("status") == "new"],
            "duplicate_papers": [p for p in all_papers if p.get("status") == "duplicate"],
            "review_papers": [p for p in all_papers if p.get("status") == "review"],
            "download_failed_papers": [p for p in all_papers if p.get("status") == "download_failed"],
            "all_papers": all_papers,
            "csv_file": management_result["summary"]["csv_path"]
        }
        update_log = generate_update_log(metadata, update_folder)
        save_update_log(update_folder, update_log)

        # Step 7: Check if there are new papers before starting infrastructure
        if not new_papers:
            logger.info("No new papers found. Skipping indexing, QA, and infrastructure startup.")

            logger.info("\n" + "=" * 60)
            logger.info("IRIS Update Cycle Completed (No new papers)")
            logger.info("=" * 60)
            logger.info(f"Update folder: {update_folder}")
            return True

        # Step 8: Start infrastructure ONLY if there are new papers
        logger.info(f"\n[Step 8] Starting infrastructure for {len(new_papers)} new papers...")
        deploy_server = DeployService(config)

        if not deploy_server.start_infrastructure():
            logger.error("Failed to start infrastructure (Milvus or vLLM models)")
            logger.error("Update cycle terminated due to infrastructure failure")
            return False

        # Step 9: Build index with new papers
        logger.info("\n[Step 9] Building index with new services...")
        papers_with_answers = []

        # 使用主集合进行增量索引
        collection_name = config["milvus"]["collection_name"]

        index_service = create_index_service_from_config(config)

        index_output_dir = update_folder / "index_storage"

        # overwrite=false 用于 Milvus 增量更新
        success = asyncio.run(index_service.chunk_and_index(
            pdf_dir=pdf_dir,
            output_dir=index_output_dir,
            collection_name=collection_name,
            overwrite=False
        ))

        if not success:
            logger.error("Index build failed, stopping infrastructure and terminating")
            deploy_server.stop_infrastructure()
            return False

        logger.info("Index created successfully")

        # Step 10: QA 处理
        logger.info("\n[Step 10] QA processing...")
        qa_service = create_qa_service_from_config(config)

        # Load questions
        questions = load_questions(config["qa"]["question_set_path"])

        # QA for each paper (using first 5 questions for summarization)
        # 使用 specific 模式按 paper_id 过滤
        for paper in new_papers:
            paper_questions = questions[:5]  # Use first 5 questions
            # 提取 arXiv ID 作为 paper_id
            paper_id = paper["entry_id"].split('/')[-1]

            answers = asyncio.run(qa_service.query_knowledge_base_with_mode(
                questions=paper_questions,
                mode="specific",
                paper_id=paper_id
            ))

            # Map answers to question keys
            answers_dict = {}
            for idx, ans in enumerate(answers):
                answers_dict[f"q{idx+1}"] = ans.get("answer", "")

            papers_with_answers.append({
                "paper": paper,
                "answers": answers_dict
            })
            logger.info(f"Generated summary for: {paper['title'][:50]}...")

        # Step 11: Generate summary and knowledge logs
        if papers_with_answers:
            logger.info("\n[Step 11] Generating summary and knowledge logs...")
            summary_md = generate_summary_markdown(papers_with_answers)
            knowledge_md = generate_knowledge_markdown(papers_with_answers)

            save_summary_log(update_folder, summary_md)
            save_knowledge_log(update_folder, knowledge_md)

            logger.info("Logs generated successfully")

            # Step 11.5: Save new papers with Q&A to SQLite database
            logger.info(f"\n[Step 11.5] Saving {len(papers_with_answers)} new papers to database...")
            for paper_data in papers_with_answers:
                paper = paper_data["paper"]
                answers = paper_data["answers"]

                # Merge paper metadata with Q&A answers
                paper_with_answers = {**paper, **answers}

                if paper_service.add_paper(paper_with_answers):
                    logger.debug(f"Saved to database: {paper['title'][:50]}...")
                else:
                    logger.warning(f"Failed to save (duplicate): {paper['title'][:50]}...")

            logger.info("Database update completed")
        else:
            logger.warning("No papers with answers, skipping log generation")

        # Step 12: Send email notification
        logger.info("\n[Step 12] Sending email notification...")
        if config["email"]["enabled"]:
            email_service = EmailService(
                sender=config["email"]["sender"],
                smtp_server=config["email"]["smtp_server"],
                smtp_port=config["email"]["smtp_port"],
                password=config["email"]["password"],
                receiver=config["email"]["receiver"],
                subject_prefix=config["email"]["subject_prefix"]
            )

            summaries_md = None
            knowledge_md = None
            if papers_with_answers:
                summaries_md = generate_summary_markdown(papers_with_answers)
                knowledge_md = generate_knowledge_markdown(papers_with_answers)

            email_success = email_service.send_update_notification(
                update_folder=update_folder,
                new_papers_count=len(new_papers),
                summaries=summaries_md,
                knowledge_log=knowledge_md
            )

            if email_success:
                logger.info("Email notification sent successfully")
            else:
                logger.warning("Email notification failed")
        else:
            logger.info("Email notifications disabled in config")

        # Step 13: Stop infrastructure
        logger.info("\n[Step 13] Stopping infrastructure...")
        deploy_server.stop_infrastructure()
        logger.info("Infrastructure stopped successfully")

        logger.info("\n" + "=" * 60)
        logger.info("IRIS Update Cycle Completed Successfully")
        logger.info("=" * 60)
        logger.info(f"Update folder: {update_folder}")
        logger.info(f"New papers: {len(new_papers)}")

        return True

    except Exception as e:
        logger.error(f"Error during update cycle: {e}", exc_info=True)
        logger.info("\n" + "=" * 60)
        logger.info("IRIS Update Cycle Failed")
        logger.info("=" * 60)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IRIS Update Cycle Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_update_cycle.py
  python run_update_cycle.py --config configs/config.yaml
  python run_update_cycle.py --log-level DEBUG
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override log level if specified
    if args.log_level:
        config["logging"]["level"] = args.log_level

    # Setup logging
    setup_logging(
        log_dir=config["logging"]["log_dir"],
        level=config["logging"]["level"]
    )

    # Run update cycle
    success = run_update_cycle(config)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()