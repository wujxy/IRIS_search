#!/usr/bin/env python3
"""
IRIS Update Cycle Orchestrator
Main script to run a complete IRIS update cycle.

Usage:
    python scripts/run_update_cycle.py
    python scripts/run_update_cycle.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.arxiv_service import ArxivService
from services.index_service import IndexService
from services.qa_service import QAService
from services.email_service import EmailService
from services.deploy_service import DeployService
from utils.helpers import (
    load_config,
    load_questions,
    setup_logging,
    create_update_folder,
    save_metadata,
    load_all_paper_entries,
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


def generate_update_log(metadata: dict, update_folder: Path) -> str:
    """
    Generate update log markdown.

    Args:
        metadata: Metadata dictionary
        update_folder: Path to update folder

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
    markdown += f"- Review papers (excluded): {len(metadata['review_papers'])}\n\n"

    if metadata['new_papers']:
        markdown += "## New Papers\n\n"
        for paper in metadata['new_papers']:
            markdown += f"- **{paper['title']}**\n"
            markdown += f"  - Authors: {', '.join(paper['authors'][:3])}\n"
            markdown += f"  - arXiv: {paper['entry_id']}\n\n"

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

        # Step 2: Load existing paper entries for duplicate detection
        logger.info("\n[Step 2] Loading existing paper entries...")
        existing_ids = load_all_paper_entries(config["storage"]["database_root"])
        logger.info(f"Found {len(existing_ids)} existing papers in database")

        # Step 3: Initialize arXiv service
        logger.info("\n[Step 3] Initializing arXiv service...")
        arxiv_config = config["arxiv"]
        arxiv_service = ArxivService(
            keywords=arxiv_config["keywords"],
            max_results=arxiv_config["max_results_per_keyword"],
            sort_by=arxiv_config["sort_by"],
            review_keywords=config["filtering"]["review_keywords"],
            exclude_reviews=config["filtering"]["exclude_reviews"]
        )

        # Step 4: Search arXiv
        logger.info("\n[Step 4] Searching arXiv for papers...")
        papers = arxiv_service.search_papers()

        # Step 5: Filter papers
        logger.info("\n[Step 5] Filtering papers...")
        filtered = arxiv_service.filter_papers(papers, existing_ids)

        # Step 6: Download PDFs
        logger.info("\n[Step 6] Downloading PDFs...")
        new_papers = filtered["new"]
        if new_papers:
            pdf_dir = update_folder / "pdfs"
            papers_with_pdf = arxiv_service.download_pdfs(new_papers, pdf_dir)

            # Filter out papers where download failed
            new_papers = [
                p for p in papers_with_pdf
                if p.get("download_success", False)
            ]
        else:
            logger.info("No new papers to download")
            papers_with_pdf = []

        # Combine all papers with status
        all_papers = filtered["new"] + filtered["duplicate"] + filtered["review"]
        all_papers = [
            p for p in all_papers
            if "download_success" not in p or p.get("download_success", True)
        ]

        # Step 7: Save metadata
        logger.info("\n[Step 7] Saving metadata...")
        save_metadata(all_papers, update_folder / "metadata.json")

        # Step 8: Generate update log
        logger.info("\n[Step 8] Generating update log...")
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_papers": len(all_papers),
            "new_papers": new_papers,
            "duplicate_papers": filtered["duplicate"],
            "review_papers": filtered["review"],
            "all_papers": all_papers
        }
        update_log = generate_update_log(metadata, update_folder)
        save_update_log(update_folder, update_log)

        # Step 8.5: 启动基础设施（Milvus + vLLM 索引模型 + vLLM QA 模型）
        logger.info("\n[Step 8.5] Starting infrastructure...")
        deploy_server = DeployService(config)

        if not deploy_server.start_infrastructure():
            logger.error("Failed to start infrastructure")
            return False

        # Step 9: Build index if there are new papers
        papers_with_answers = []
        if new_papers:
            logger.info("\n[Step 9] Building index with UltraRAG...")

            # 生成批次命名的 collection_name
            collection_name = f"update_{datetime.now().strftime('%Y_%m_%d_%H%M')}"

            index_service = IndexService(
                ultrarag_path=config["ultrarag"]["ultrarag_path"],
                embedding_model=config["models"]["embedding_model_path"],
                collection_name=collection_name
            )

            index_output_dir = update_folder / "index_storage"

            # overwrite=false 用于 Milvus 增量更新
            success = index_service.chunk_and_index(
                pdf_dir=pdf_dir,
                output_dir=index_output_dir,
                collection_name=collection_name,
                overwrite=False
            )

            if success:
                index_info = index_service.get_index_info(index_output_dir)
                logger.info(f"Index created: {index_info['chunk_count']} chunks")

                # Step 10: QA 处理（vLLM 模型已在 Step 8.5 中同时启动）
                logger.info("\n[Step 10] QA processing...")
                # vLLM 模型（索引和 QA）已在 Step 8.5 中同时启动
                qa_service = QAService(
                    ultrarag_path=config["ultrarag"]["ultrarag_path"],
                    embedding_model=config["models"]["embedding_model_path"],
                    reranker_model=config["models"]["reranker_model_path"],
                    generation_model=config["models"]["llm_model_path"],
                    vllm_base_url=config["models"]["vllm"]["base_url"],
                    index_vllm_base_url=config["models"]["vllm"]["index"]["base_url"],
                    served_model_name=config["models"]["vllm"]["served_model_name"],
                    system_prompt=config["qa"]["system_prompt"],
                    collection_name=collection_name
                )

                # Load questions
                questions = load_questions(config["qa"]["question_set_path"])

                # QA for each paper (using first 5 questions for summarization)
                chunks_path = Path(index_info["chunks_file"])
                index_path = Path(index_info["index_file"]) if index_info["index_file"] else None

                if chunks_path.exists():
                    for paper in new_papers:
                        paper_questions = questions[:5]  # Use first 5 questions
                        answers = qa_service.query_knowledge_base_batch(
                            questions=paper_questions,
                            chunks_path=chunks_path,
                            collection_name=collection_name
                        )

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
                        logger.info("\n[Step 11] Generating summary and knowledge logs...")
                        summary_md = generate_summary_markdown(papers_with_answers)
                        knowledge_md = generate_knowledge_markdown(papers_with_answers)

                        save_summary_log(update_folder, summary_md)
                        save_knowledge_log(update_folder, knowledge_md)

                        logger.info("Logs generated successfully")
                    else:
                        logger.warning("Index files not found, skipping QA")
                else:
                    logger.warning("vLLM service not available, skipping QA")
            else:
                logger.error("Index build failed, skipping QA")
        else:
            logger.info("No new papers, skipping indexing and QA")

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

        # Step 11.5: 停止基础设施
        logger.info("\n[Step 11.5] Stopping infrastructure...")
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
