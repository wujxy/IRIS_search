"""
arXiv Search Service for IRIS
Handles arXiv paper search and PDF download with filtering capabilities.
"""

import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

import arxiv


logger = logging.getLogger(__name__)


class ArxivService:
    """Service for searching and downloading papers from arXiv."""

    def __init__(
        self,
        keywords: List[str],
        max_results: int = 20,
        sort_by: str = "SubmittedDate",
        review_keywords: List[str] = None,
        exclude_reviews: bool = True,
        last_update_time: Optional[str] = None,
        database_root: Optional[str] = None
    ):
        """
        Initialize arXiv service.

        Args:
            keywords: List of keywords to search for
            max_results: Maximum results per keyword
            sort_by: Sort criterion (SubmittedDate, LastUpdatedDate, Relevance)
            review_keywords: Keywords that indicate review papers
            exclude_reviews: Whether to filter out review papers
            last_update_time: Last update time for incremental search
            database_root: Database root path for loading history
        """
        self.keywords = keywords
        self.max_results = max_results
        self.sort_by = self._get_sort_criterion(sort_by)
        self.review_keywords = review_keywords or ["review", "survey", "overview"]
        self.exclude_reviews = exclude_reviews
        self.last_update_time = last_update_time
        self.database_root = database_root

    def _get_sort_criterion(self, sort_by: str) -> arxiv.SortCriterion:
        """Convert string to arxiv.SortCriterion enum."""
        sort_map = {
            "SubmittedDate": arxiv.SortCriterion.SubmittedDate,
            "LastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "Relevance": arxiv.SortCriterion.Relevance
        }
        return sort_map.get(sort_by, arxiv.SortCriterion.SubmittedDate)

    def _get_last_update_time(self) -> str:
        """
        Get last update time for incremental search.

        Returns:
            String timestamp for search start time
        """
        if self.last_update_time:
            return self.last_update_time

        # Load from database
        if not self.database_root:
            # Default: search papers from last 24 hours
            return (datetime.now() - timedelta(hours=24)).strftime("%Y%m%d%H%M%S")

        database_root = Path(self.database_root)
        update_folders = sorted(database_root.glob("update_*"), reverse=True)
        if update_folders:
            latest_folder = update_folders[0]
            metadata_file = latest_folder / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("timestamp", "")
        return ""

    def search_papers(self) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching configured keywords.

        Returns:
            List of paper metadata dictionaries
        """
        logger.info(f"Searching arXiv with keywords: {self.keywords}")

        # Build query with OR logic
        query_parts = []
        for item in self.keywords:
            if " " in item:
                query_parts.append('"' + item + '"')
            else:
                query_parts.append(item)
        query = " OR ".join(query_parts)

        logger.debug(f"ArXiv query: {query}")

        # Create search
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=self.sort_by
        )

        papers = []
        seen_ids = set()

        # Iterate through results
        for result in search.results():
            # Skip duplicates in current search
            if result.entry_id in seen_ids:
                continue
            seen_ids.add(result.entry_id)

            paper = self._convert_result_to_dict(result)
            papers.append(paper)

        logger.info(f"Found {len(papers)} papers from arXiv")
        return papers

    def search_papers_incremental(self, start_time: str = None) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers after a specific time (incremental search).

        Args:
            start_time: Start time in format %Y%m%d%H%M%S. If None, uses last update.

        Returns:
            List of paper metadata dictionaries
        """
        start_time = start_time or self._get_last_update_time()
        logger.info(f"Searching arXiv for papers since {start_time}")

        # Parse start_time to datetime for comparison
        start_dt = None
        if start_time:
            try:
                start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                try:
                    start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    logger.warning(f"Invalid start_time format: {start_time}")

        query_parts = []
        for item in self.keywords:
            if " " in item:
                query_parts.append('"' + item + '"')
            else:
                query_parts.append(item)
        query = " OR ".join(query_parts)

        search = arxiv.Search(
            query=query,
            max_results=self.max_results * 2,  # Search more for retry
            sort_by=self.sort_by
        )

        papers = []
        seen_ids = set()

        for result in search.results():
            if result.entry_id in seen_ids:
                continue
            seen_ids.add(result.entry_id)

            paper = self._convert_result_to_dict(result)

            # Filter by actual update/published time if start_time is provided
            if start_dt:
                paper_published = None
                paper_updated = None

                if paper.get("published"):
                    try:
                        paper_published = datetime.fromisoformat(paper["published"].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        pass

                if paper.get("updated"):
                    try:
                        paper_updated = datetime.fromisoformat(paper["updated"].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        pass

                # Check if paper was updated or published after start_time
                is_newer = False
                if paper_updated and paper_updated >= start_dt:
                    is_newer = True
                elif paper_published and paper_published >= start_dt:
                    is_newer = True

                if not is_newer:
                    logger.debug(f"Skipping paper (too old): {paper['title'][:50]}...")
                    continue

            papers.append(paper)

        logger.info(f"Found {len(papers)} papers from arXiv after time filter")
        return papers

    def filter_papers(
        self,
        papers: List[Dict[str, Any]],
        existing_ids: Set[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filter papers by review status and duplicates.

        Args:
            papers: List of paper metadata
            existing_ids: Set of existing entry_ids to check for duplicates

        Returns:
            Dictionary with categorized papers:
            {
                "new": [...],
                "duplicate": [...],
                "review": [...]
            }
        """
        existing_ids = existing_ids or set()

        filtered = {
            "new": [],
            "duplicate": [],
            "review": []
        }

        for paper in papers:
            # Check for duplicates
            if paper["entry_id"] in existing_ids:
                paper["status"] = "duplicate"
                filtered["duplicate"].append(paper)
                logger.debug(f"Duplicate paper: {paper['title'][:50]}...")
                continue

            # Check for review papers
            if self.exclude_reviews and self._is_review_paper(paper):
                paper["status"] = "review"
                filtered["review"].append(paper)
                logger.debug(f"Review paper: {paper['title'][:50]}...")
                continue

            # New paper
            paper["status"] = "new"
            filtered["new"].append(paper)

        logger.info(
            f"Filtering results: {len(filtered['new'])} new, "
            f"{len(filtered['duplicate'])} duplicates, "
            f"{len(filtered['review'])} review papers"
        )

        return filtered

    def _is_review_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Check if a paper is a review/survey/overview paper.

        Args:
            paper: Paper metadata dictionary

        Returns:
            True if paper is a review paper
        """
        text = (paper["title"] + " " + paper["summary"]).lower()

        for keyword in self.review_keywords:
            if keyword.lower() in text:
                return True

        return False

    def _convert_result_to_dict(self, arxiv_result) -> Dict[str, Any]:
        """
        Convert arXiv Result object to dictionary.

        Args:
            arxiv_result: arxiv.Result object

        Returns:
            Dictionary containing paper metadata
        """
        return {
            "entry_id": arxiv_result.entry_id,
            "updated": arxiv_result.updated.isoformat() if arxiv_result.updated else None,
            "published": arxiv_result.published.isoformat() if arxiv_result.published else None,
            "title": arxiv_result.title,
            "authors": [str(author) for author in arxiv_result.authors],
            "summary": arxiv_result.summary.replace('\n', ' ').strip(),
            "comment": arxiv_result.comment,
            "journal_ref": arxiv_result.journal_ref,
            "doi": arxiv_result.doi,
            "primary_category": arxiv_result.primary_category,
            "categories": arxiv_result.categories,
            "pdf_url": arxiv_result.pdf_url
        }

    def download_pdf(
        self,
        paper: Dict[str, Any],
        save_dir: Path,
        filename: Optional[str] = None
    ) -> tuple[Optional[Path], bool]:
        """
        Download PDF for a paper.

        Args:
            paper: Paper metadata dictionary
            save_dir: Directory to save PDF
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Tuple of (file_path, success) where success is True if download succeeded
        """
        if filename is None:
            # Generate filename from entry_id
            filename = paper["entry_id"].split('/')[-1] + ".pdf"

        save_path = save_dir / filename

        try:
            # Download PDF using requests
            import requests
            response = requests.get(paper["pdf_url"], stream=True, timeout=30)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded PDF: {paper['title'][:50]}... -> {filename}")
            return save_path, True

        except Exception as e:
            logger.error(f"Failed to download PDF for {paper['title'][:50]}...: {e}")
            return None, False

    def download_pdfs(
        self,
        papers: List[Dict[str, Any]],
        save_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Download PDFs for multiple papers.

        Args:
            papers: List of paper metadata dictionaries
            save_dir: Directory to save PDFs

        Returns:
            List of papers with updated pdf_path field
        """
        logger.info(f"Downloading PDFs for {len(papers)} papers...")

        for paper in papers:
            if paper["status"] != "new":
                # Skip duplicates and review papers
                continue

            pdf_path, success = self.download_pdf(paper, save_dir)
            paper["pdf_path"] = str(pdf_path) if pdf_path else None
            paper["download_success"] = success

        successful = sum(1 for p in papers if p.get("download_success", False))
        logger.info(f"Successfully downloaded {successful}/{len(papers)} PDFs")

        return papers

    def download_pdfs_with_retry(
        self,
        papers: List[Dict[str, Any]],
        save_dir: Path,
        target_count: int = None
    ) -> List[Dict[str, Any]]:
        """
        Download PDFs with retry mechanism to achieve target count.

        Args:
            papers: List of paper metadata
            save_dir: Directory to save PDFs
            target_count: Target number of successfully downloaded papers

        Returns:
            Updated list of papers with download status
        """
        target_count = target_count or self.max_results
        downloaded_papers = []
        failed_downloads = []
        checked_ids = set()

        # Safety limits to prevent infinite loops
        MAX_RETRIES = 3
        MAX_TOTAL_PAPERS_TO_CHECK = target_count * 5
        retry_count = 0
        total_checked = 0

        while len(downloaded_papers) < target_count and retry_count < MAX_RETRIES:
            new_in_batch = False

            for paper in papers:
                # Safety limit
                if total_checked >= MAX_TOTAL_PAPERS_TO_CHECK:
                    logger.warning(f"Reached maximum papers to check limit ({MAX_TOTAL_PAPERS_TO_CHECK})")
                    break

                # Skip already checked papers
                if paper["entry_id"] in checked_ids:
                    continue
                checked_ids.add(paper["entry_id"])
                total_checked += 1

                # Papers passed to this function are already filtered as "new" from manage_papers()
                logger.info(f"Downloading PDF for: {paper['title'][:60]}...")

                pdf_path, success = self.download_pdf(paper, save_dir)
                paper["pdf_path"] = str(pdf_path) if pdf_path else None
                paper["download_success"] = success

                if success:
                    paper["download_status"] = "download"
                    paper["status"] = "new"
                    downloaded_papers.append(paper)
                    new_in_batch = True
                    logger.info(f"Successfully downloaded: {paper['title'][:40]}...")
                else:
                    paper["download_status"] = "download_failed"
                    paper["status"] = "download_failed"
                    failed_downloads.append(paper)
                    logger.warning(f"Failed to download PDF: {paper['title'][:40]}...")

            # If no new papers or reached max count, stop
            if not new_in_batch:
                logger.info("No new papers in this batch, stopping...")
                break

            # If still not enough, search for more papers
            if len(downloaded_papers) < target_count:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    logger.warning(f"Max retries ({MAX_RETRIES}) reached")
                    break
                logger.info(f"Retry {retry_count}/{MAX_RETRIES}: Need more papers ({len(downloaded_papers)}/{target_count}), searching...")
                papers = self.search_papers_incremental()

        # Combine all processed papers
        all_papers = downloaded_papers + failed_downloads
        logger.info(f"Downloaded {len(downloaded_papers)} papers, failed {len(failed_downloads)}")

        return all_papers

    def _load_existing_ids(self) -> set:
        """
        Load existing paper IDs from database.

        Returns:
            Set of existing entry_ids
        """
        if not self.database_root:
            return set()

        database_root = Path(self.database_root)
        entry_ids = set()

        for update_folder in database_root.glob("update_*"):
            metadata_file = update_folder / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for paper in data.get("all_papers", []):
                            entry_id = paper.get("entry_id")
                            if entry_id:
                                entry_ids.add(entry_id)
                except (json.JSONDecodeError, KeyError, Exception) as e:
                    # Skip corrupted metadata files
                    logger.warning(f"Skipping corrupted metadata file {metadata_file}: {e}")
                    continue

        return entry_ids

    def save_to_csv(
        self,
        papers: List[Dict[str, Any]],
        csv_path: Path
    ) -> None:
        """
        Save paper information to CSV file with all required fields.

        CSV columns: entry_id, title, authors, published, updated, summary, comment, journal_ref,
                     doi, primary_category, categories, status, download_status, pdf_url

        Args:
            papers: List of paper metadata dictionaries
            csv_path: Path to save CSV file
        """
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "entry_id", "title", "authors", "published", "updated", "summary", "comment",
                "journal_ref", "doi", "primary_category", "categories",
                "status", "download_status", "pdf_url"
            ])

            # Rows
            for paper in papers:
                writer.writerow([
                    paper.get("entry_id", ""),
                    paper.get("title", ""),
                    "; ".join(paper.get("authors", [])[:3]),
                    paper.get("published", ""),
                    paper.get("updated", ""),
                    paper.get("summary", ""),
                    paper.get("comment", ""),
                    paper.get("journal_ref", ""),
                    paper.get("doi", ""),
                    paper.get("primary_category", ""),
                    paper.get("categories", ""),
                    paper.get("status", ""),
                    paper.get("download_status", ""),
                    paper.get("pdf_url", "")
                ])

        logger.info(f"Saved {len(papers)} papers to CSV: {csv_path}")

    def manage_papers(
        self,
        save_dir: Path,
        target_count: int = None,
        csv_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Complete paper management workflow: search, filter, download, export.

        Args:
            save_dir: Directory to save PDFs
            target_count: Target number of papers to process
            csv_dir: Directory to save CSV file (defaults to save_dir.parent)

        Returns:
            Dictionary containing:
                - papers: All processed papers with status
                - downloaded: Successfully downloaded papers
                - summary: Statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Paper Management Workflow")
        logger.info("=" * 60)

        # Step 1: Search arXiv (incremental)
        logger.info("\n[Step 1] Searching arXiv incrementally...")
        papers = self.search_papers_incremental()

        # Step 2: Filter papers (determine new/duplicate/review)
        logger.info("\n[Step 2] Filtering papers...")
        existing_ids = self._load_existing_ids()
        filtered = self.filter_papers(papers, existing_ids)

        # Step 3: Download PDFs for new papers with retry
        logger.info("\n[Step 3] Downloading PDFs with retry...")
        processed_papers = self.download_pdfs_with_retry(
            papers=filtered["new"],
            save_dir=save_dir,
            target_count=target_count
        )

        # Combine all processed papers
        all_papers = processed_papers + filtered["duplicate"] + filtered["review"]

        # Step 4: Export to CSV (save to parent directory of save_dir)
        csv_dir = csv_dir or save_dir.parent
        csv_path = csv_dir / "papers.csv"
        self.save_to_csv(all_papers, csv_path)

        # Summary
        summary = {
            "total": len(all_papers),
            "new": len(filtered["new"]),
            "duplicate": len(filtered["duplicate"]),
            "review": len(filtered["review"]),
            "downloaded": len(processed_papers),
            "download_failed": len([p for p in processed_papers if p.get("status") == "download_failed"]),
            "csv_path": str(csv_path)
        }

        logger.info("\n" + "=" * 60)
        logger.info("Paper Management Summary:")
        logger.info(f"  Total papers: {summary['total']}")
        logger.info(f"  New: {summary['new']}")
        logger.info(f"  Duplicate: {summary['duplicate']}")
        logger.info(f"  Review: {summary['review']}")
        logger.info(f"  Downloaded: {summary['downloaded']}")
        logger.info(f"  Failed: {summary['download_failed']}")
        logger.info(f"  CSV file: {summary['csv_path']}")
        logger.info("=" * 60)

        return {
            "papers": all_papers,
            "downloaded": processed_papers,
            "summary": summary
        }
