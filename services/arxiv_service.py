"""
arXiv Search Service for IRIS
Handles arXiv paper search and PDF download with filtering capabilities.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

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
        exclude_reviews: bool = True
    ):
        """
        Initialize arXiv service.

        Args:
            keywords: List of keywords to search for
            max_results: Maximum results per keyword
            sort_by: Sort criterion (SubmittedDate, LastUpdatedDate, RelevanceDate)
            review_keywords: Keywords that indicate review papers
            exclude_reviews: Whether to filter out review papers
        """
        self.keywords = keywords
        self.max_results = max_results
        self.sort_by = self._get_sort_criterion(sort_by)
        self.review_keywords = review_keywords or ["review", "survey", "overview"]
        self.exclude_reviews = exclude_reviews

    def _get_sort_criterion(self, sort_by: str) -> arxiv.SortCriterion:
        """Convert string to arxiv.SortCriterion enum."""
        sort_map = {
            "SubmittedDate": arxiv.SortCriterion.SubmittedDate,
            "LastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "Relevance": arxiv.SortCriterion.Relevance
        }
        return sort_map.get(sort_by, arxiv.SortCriterion.SubmittedDate)

    def search_papers(self) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching the configured keywords.

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
            "pdf_url": arxiv_result.pdf_url,
            "status": "new"
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
            save_dir: Directory to save the PDF
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
