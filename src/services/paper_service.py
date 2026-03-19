"""
Paper Service for IRIS
Manages SQLite database for paper metadata storage and queries.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PaperService:
    """Service for managing paper metadata in SQLite database."""

    def __init__(self, db_path: str):
        """
        Initialize paper service with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path).absolute()
        self._initialize_database()
        logger.info(f"Paper service initialized with database: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create papers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT UNIQUE NOT NULL,
                    paper_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authors TEXT,
                    published TEXT,
                    updated TEXT,
                    summary TEXT,
                    comment TEXT,
                    journal_ref TEXT,
                    doi TEXT,
                    primary_category TEXT,
                    categories TEXT,
                    status TEXT,
                    download_status TEXT,
                    pdf_url TEXT,
                    pdf_path TEXT,
                    update_folder TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_paper_id ON papers(paper_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entry_id ON papers(entry_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_published ON papers(published DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_update_folder ON papers(update_folder)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON papers(status)
            """)

            conn.commit()
            logger.info("Database tables and indexes initialized")

    def add_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Add a single paper to database.

        Args:
            paper: Dictionary containing paper metadata

        Returns:
            True if added successfully, False if duplicate
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Extract paper_id from entry_id
                entry_id = paper.get('entry_id', '')
                paper_id = entry_id.split('/')[-1].split('v')[0] if entry_id else ''

                cursor.execute("""
                    INSERT INTO papers (
                        entry_id, paper_id, title, authors, published, updated,
                        summary, comment, journal_ref, doi, primary_category,
                        categories, status, download_status, pdf_url, pdf_path, update_folder
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id,
                    paper_id,
                    paper.get('title', ''),
                    json.dumps(paper.get('authors', [])),
                    paper.get('published'),
                    paper.get('updated'),
                    paper.get('summary', ''),
                    paper.get('comment'),
                    paper.get('journal_ref'),
                    paper.get('doi'),
                    paper.get('primary_category'),
                    json.dumps(paper.get('categories', [])),
                    paper.get('status'),
                    paper.get('download_status'),
                    paper.get('pdf_url'),
                    paper.get('pdf_path'),
                    paper.get('update_folder', '')
                ))

                conn.commit()
                logger.debug(f"Added paper: {paper_id}")
                return True

        except sqlite3.IntegrityError as e:
            # Unique constraint violation (duplicate entry_id)
            logger.debug(f"Duplicate paper skipped: {paper.get('entry_id')}")
            return False

        except Exception as e:
            logger.error(f"Error adding paper: {e}")
            return False

    def add_papers_batch(self, papers: List[Dict[str, Any]], update_folder_name: str = None) -> int:
        """
        Add multiple papers to database in batch.

        Args:
            papers: List of paper dictionaries
            update_folder_name: Update folder name to associate with papers

        Returns:
            Number of papers added
        """
        added_count = 0

        for paper in papers:
            # Add update_folder if provided
            if update_folder_name:
                paper['update_folder'] = update_folder_name

            if self.add_paper(paper):
                added_count += 1

        logger.info(f"Batch added {added_count}/{len(papers)} papers")
        return added_count

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get paper by paper_id.

        Args:
            paper_id: Paper ID (e.g., "2403.15570")

        Returns:
            Paper dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM papers WHERE paper_id = ?
                LIMIT 1
            """, (paper_id,))

            row = cursor.fetchone()

            if row:
                return self._row_to_dict(cursor, row)
            return None

    def get_paper_by_entry_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get paper by entry_id.

        Args:
            entry_id: Full entry ID (e.g., "http://arxiv.org/abs/2403.15570v1")

        Returns:
            Paper dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM papers WHERE entry_id = ?
                LIMIT 1
            """, (entry_id,))

            row = cursor.fetchone()

            if row:
                return self._row_to_dict(cursor, row)
            return None

    def list_papers(
        self,
        limit: int = None,
        offset: int = 0,
        order_by: str = "published",
        reverse: bool = True,
        status: str = None
    ) -> List[Dict[str, Any]]:
        """
        List papers with optional sorting and pagination.

        Args:
            limit: Maximum number of papers to return
            offset: Number of papers to skip
            order_by: Column to order by (default: "published")
            reverse: Sort in descending order if True
            status: Filter by status (e.g., "new", "review")

        Returns:
            List of paper dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build query
            query = "SELECT * FROM papers"
            params = []

            # Add status filter if specified
            if status:
                query += " WHERE status = ?"
                params.append(status)

            # Add order by
            if reverse:
                query += f" ORDER BY {order_by} DESC"
            else:
                query += f" ORDER BY {order_by}"

            # Add limit and offset
            if limit:
                query += " LIMIT ?"
                params.append(limit)
                if offset:
                    query += " OFFSET ?"
                    params.append(offset)
            elif offset:
                query += " OFFSET ?"
                params.append(offset)

            cursor.execute(query, tuple(params) if params else ())

            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows]

    def search_papers(
        self,
        keyword: str = None,
        category: str = None,
        status: str = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search papers by keyword, category, and/or status.

        Args:
            keyword: Search keyword in title or summary
            category: Filter by primary_category
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of matching paper dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build query with filters
            conditions = []
            params = []

            if keyword:
                conditions.append("(title LIKE ? OR summary LIKE ?)")
                keyword_pattern = f"%{keyword}%"
                params.extend([keyword_pattern, keyword_pattern])

            if category:
                conditions.append("primary_category = ?")
                params.append(category)

            if status:
                conditions.append("status = ?")
                params.append(status)

            # Build WHERE clause
            if conditions:
                query = "SELECT * FROM papers WHERE " + " AND ".join(conditions)
            else:
                query = "SELECT * FROM papers"

            # Order by published date (newest first)
            query += " ORDER BY published DESC"

            # Add limit
            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, tuple(params) if params else ())

            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows]

    def get_stats(self) -> Dict[str, int]:
        """
        Get database statistics.

        Returns:
            Dictionary with total counts by status
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM papers")
            total = cursor.fetchone()[0]

            # Get counts by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM papers
                GROUP BY status
            """)

            stats = {"total": total}
            for row in cursor.fetchall():
                stats[row["status"]] = row["count"]

            return stats

    def get_paper_id_from_entry_id(self, entry_id: str) -> str:
        """
        Extract paper_id from entry_id.

        Args:
            entry_id: Full entry ID (e.g., "http://arxiv.org/abs/2403.15570v1")

        Returns:
            Paper ID (e.g., "2403.15570")
        """
        return entry_id.split('/')[-1].split('v')[0] if entry_id else ''

    def get_recent_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recently published papers.

        Args:
            limit: Number of papers to return

        Returns:
            List of recent paper dictionaries
        """
        return self.list_papers(limit=limit, order_by="published", reverse=True)

    def _row_to_dict(self, cursor: sqlite3.Cursor, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to dictionary with proper type conversion."""
        result = {}

        for key in row.keys():
            value = row[key]

            # Parse JSON fields
            if key in ['authors', 'categories'] and value:
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = []
            else:
                result[key] = value

        return result
