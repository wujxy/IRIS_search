"""
IRIS Update Orchestrator

This module orchestrates the complete IRIS update cycle:
1. Fetch new papers from arXiv
2. Download and filter papers
3. Build vector index
4. Run QA summarization
5. Send notifications

Extracted from scripts/run_update_cycle.py for better code organization.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import get_config
from src.common import ensure_dir
from src.exceptions import ServiceError

# Import services
from services.arxiv_service import ArxivService
from services.email_service import EmailService
from services.paper_service import PaperService
from services.deploy_service import DeployService
from core.index_service import create_index_service_from_config
from core.qa_service import create_qa_service_from_config

# Import helpers (will be migrated later)
from utils.helpers import (
    load_questions,
    create_update_folder,
    save_metadata,
    save_update_log,
    save_summary_log,
    save_knowledge_log,
)

logger = logging.getLogger(__name__)


class UpdateOrchestrator:
    """
    Orchestrates the complete IRIS update cycle.

    This class coordinates all services involved in the update process:
    - Paper fetching from arXiv
    - Index building
    - QA processing
    - Notifications

    Usage:
        orchestrator = UpdateOrchestrator(config)
        success = orchestrator.run_cycle()
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize the orchestrator.

        Args:
            config: Configuration dictionary (uses default if None)
        """
        self.config = config or get_config().config
        self.update_folder: Optional[Path] = None
        self._deploy_server: Optional[DeployService] = None

    def run_cycle(self) -> bool:
        """
        Run a complete IRIS update cycle.

        Returns:
            True if cycle completed successfully, False otherwise
        """
        logger.info("=" * 60)
        logger.info("Starting IRIS Update Cycle")
        logger.info("=" * 60)

        try:
            # Step 1: Create update folder
            self.update_folder = create_update_folder(self.config["storage"]["database_root"])
            logger.info(f"Update folder: {self.update_folder}")

            # Step 2: Load existing papers
            paper_service = PaperService(self.config["storage"]["paper_db_path"])
            existing_ids = paper_service.get_all_entry_ids()
            logger.info(f"Found {len(existing_ids)} existing papers")

            # Step 3: Fetch and manage papers
            all_papers, downloaded_papers, new_papers = self._manage_papers(existing_ids)

            # Step 4: Save metadata and logs
            metadata = self._generate_metadata(all_papers, downloaded_papers)
            self._save_update_logs(metadata)

            # Step 5: Process new papers if any
            if not new_papers:
                logger.info("No new papers found. Update cycle completed.")
                return True

            # Step 6: Start infrastructure
            if not self._start_infrastructure():
                return False

            # Step 7: Build index
            if not self._build_index(new_papers):
                self._stop_infrastructure()
                return False

            # Step 8: QA processing
            papers_with_answers = self._run_qa_processing(new_papers)

            # Step 9: Save to database
            self._save_to_database(paper_service, papers_with_answers)

            # Step 10: Send notification
            self._send_notification(new_papers, papers_with_answers)

            # Step 11: Stop infrastructure
            self._stop_infrastructure()

            logger.info("=" * 60)
            logger.info("IRIS Update Cycle Completed Successfully")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"Error during update cycle: {e}", exc_info=True)
            self._stop_infrastructure()
            return False

    def _manage_papers(self, existing_ids: list[str]) -> tuple:
        """Fetch and manage papers from arXiv."""
        logger.info("\n[Step 3] Managing papers from arXiv...")

        arxiv_config = self.config["arxiv"]
        arxiv_service = ArxivService(
            keywords=arxiv_config["keywords"],
            max_results=arxiv_config["max_results_per_keyword"],
            sort_by=arxiv_config["sort_by"],
            review_keywords=self.config["filtering"]["review_keywords"],
            exclude_reviews=self.config["filtering"]["exclude_reviews"],
            database_root=self.config["storage"]["database_root"],
        )

        pdf_dir = self.update_folder / "pdfs"
        result = arxiv_service.manage_papers(
            save_dir=pdf_dir,
            target_count=arxiv_config["max_results_per_keyword"],
            existing_ids=existing_ids,
        )

        all_papers = result["papers"]
        downloaded_papers = result["downloaded"]
        new_papers = [p for p in all_papers if p.get("status") == "new"]

        logger.info(f"Found {len(new_papers)} new papers")
        return all_papers, downloaded_papers, new_papers

    def _generate_metadata(self, all_papers: list, downloaded_papers: list) -> dict:
        """Generate metadata for the update cycle."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_papers": len(all_papers),
            "new_papers": [p for p in all_papers if p.get("status") == "new"],
            "duplicate_papers": [p for p in all_papers if p.get("status") == "duplicate"],
            "review_papers": [p for p in all_papers if p.get("status") == "review"],
            "download_failed_papers": [p for p in all_papers if p.get("status") == "download_failed"],
        }

    def _save_update_logs(self, metadata: dict) -> None:
        """Save metadata and update logs."""
        save_metadata(metadata.get("all_papers", []), self.update_folder / "metadata.json")

        update_log = self._generate_update_log(metadata)
        save_update_log(self.update_folder, update_log)

    def _generate_update_log(self, metadata: dict) -> str:
        """Generate update log markdown."""
        markdown = "# IRIS Update Log\n\n"
        markdown += f"Timestamp: {metadata['timestamp']}\n\n"
        markdown += "---\n\n## Summary\n\n"
        markdown += f"- Total papers: {metadata['total_papers']}\n"
        markdown += f"- New papers: {len(metadata['new_papers'])}\n"
        markdown += f"- Duplicates: {len(metadata['duplicate_papers'])}\n"
        markdown += f"- Reviews excluded: {len(metadata['review_papers'])}\n"
        return markdown

    def _start_infrastructure(self) -> bool:
        """Start Milvus and vLLM infrastructure."""
        logger.info("\n[Step 6] Starting infrastructure...")
        self._deploy_server = DeployService(self.config)

        if not self._deploy_server.start_infrastructure():
            logger.error("Failed to start infrastructure")
            return False
        return True

    def _stop_infrastructure(self) -> None:
        """Stop infrastructure."""
        if self._deploy_server:
            logger.info("\nStopping infrastructure...")
            self._deploy_server.stop_infrastructure()

    def _build_index(self, new_papers: list) -> bool:
        """Build vector index for new papers."""
        logger.info("\n[Step 7] Building index...")

        index_service = create_index_service_from_config(self.config)
        pdf_dir = self.update_folder / "pdfs"
        output_dir = self.update_folder / "index_storage"

        success = asyncio.run(
            index_service.chunk_and_index(
                pdf_dir=pdf_dir,
                output_dir=output_dir,
                collection_name=self.config["milvus"]["collection_name"],
                overwrite=False,
            )
        )

        if success:
            logger.info("Index created successfully")
        return success

    def _run_qa_processing(self, new_papers: list) -> list:
        """Run QA processing for new papers."""
        logger.info("\n[Step 8] QA processing...")

        qa_service = create_qa_service_from_config(self.config)
        questions = load_questions(self.config["qa"]["question_set_path"])
        papers_with_answers = []

        for paper in new_papers:
            paper_id = paper["entry_id"].split("/")[-1]
            paper_questions = questions[:5]

            answers = asyncio.run(
                qa_service.query_knowledge_base_with_mode(
                    questions=paper_questions, mode="specific", paper_id=paper_id
                )
            )

            answers_dict = {f"q{idx + 1}": ans.get("answer", "") for idx, ans in enumerate(answers)}
            papers_with_answers.append({"paper": paper, "answers": answers_dict})

            logger.info(f"Generated summary for: {paper['title'][:50]}...")

        return papers_with_answers

    def _save_to_database(self, paper_service: PaperService, papers_with_answers: list) -> None:
        """Save papers with answers to database."""
        if not papers_with_answers:
            return

        logger.info(f"\n[Step 9] Saving {len(papers_with_answers)} papers to database...")

        for paper_data in papers_with_answers:
            paper_with_answers = {**paper_data["paper"], **paper_data["answers"]}
            paper_service.add_paper(paper_with_answers)

        logger.info("Database update completed")

        # Generate and save summary/knowledge logs
        summary_md = self._generate_summary_markdown(papers_with_answers)
        knowledge_md = self._generate_knowledge_markdown(papers_with_answers)

        save_summary_log(self.update_folder, summary_md)
        save_knowledge_log(self.update_folder, knowledge_md)

    def _generate_summary_markdown(self, papers_with_answers: list) -> str:
        """Generate summary markdown from papers with answers."""
        markdown = "# Paper Summaries\n\n"
        markdown += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"

        for paper_data in papers_with_answers:
            paper = paper_data["paper"]
            answers = paper_data.get("answers", {})

            markdown += f"## {paper['title']}\n\n"
            markdown += f"**Authors:** {', '.join(paper['authors'][:5])}\n\n"
            markdown += f"**arXiv ID:** {paper['entry_id']}\n\n"
            markdown += "### Key Information\n\n"

            for q_key, q_text in [
                ("q1", "Main Problem"),
                ("q2", "Key Contributions"),
                ("q3", "Methods"),
                ("q4", "Important Concepts"),
                ("q5", "Research Directions"),
            ]:
                answer = answers.get(q_key, "N/A")
                markdown += f"**{q_text}:**\n{answer}\n\n"

            markdown += "---\n\n"

        return markdown

    def _generate_knowledge_markdown(self, papers_with_answers: list) -> str:
        """Generate knowledge extraction markdown."""
        markdown = "# Knowledge Extraction\n\n"
        markdown += f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"

        for paper_data in papers_with_answers:
            paper = paper_data["paper"]
            answers = paper_data.get("answers", {})

            markdown += f"## {paper['title']}\n\n"
            markdown += "### Key Contributions\n\n"
            markdown += f"{answers.get('q2', '')}\n\n"
            markdown += "### Methods and Techniques\n\n"
            markdown += f"{answers.get('q3', '')}\n\n"
            markdown += "---\n\n"

        return markdown

    def _send_notification(self, new_papers: list, papers_with_answers: list) -> None:
        """Send email notification if enabled."""
        logger.info("\n[Step 10] Sending notification...")

        if not self.config["email"]["enabled"]:
            logger.info("Email notifications disabled")
            return

        email_service = EmailService(
            sender=self.config["email"]["sender"],
            smtp_server=self.config["email"]["smtp_server"],
            smtp_port=self.config["email"]["smtp_port"],
            password=self.config["email"]["password"],
            receiver=self.config["email"]["receiver"],
            subject_prefix=self.config["email"]["subject_prefix"],
        )

        summaries_md = self._generate_summary_markdown(papers_with_answers) if papers_with_answers else None
        knowledge_md = self._generate_knowledge_markdown(papers_with_answers) if papers_with_answers else None

        email_success = email_service.send_update_notification(
            update_folder=self.update_folder,
            new_papers_count=len(new_papers),
            summaries=summaries_md,
            knowledge_log=knowledge_md,
        )

        if email_success:
            logger.info("Email notification sent")
        else:
            logger.warning("Email notification failed")


