"""
Task Service for IRIS Scheduler

Manages SQLite database for task state tracking and history.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .task_models import ScheduledTask, TaskStatus, TaskType, TriggerType

logger = logging.getLogger(__name__)


class TaskService:
    """Service for managing task records in database."""

    def __init__(self, db_path: str):
        """
        Initialize task service with SQLite database.

        Args:
            db_path: Path to SQLite database file (same as papers database)
        """
        self.db_path = Path(db_path).absolute()
        self._initialize_database()
        logger.info(f"Task service initialized with database: {self.db_path}")

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
        """Create scheduled_tasks table if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create scheduled_tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    scheduled_time TIMESTAMP NOT NULL,
                    started_time TIMESTAMP,
                    completed_time TIMESTAMP,
                    trigger_type TEXT NOT NULL,
                    trigger_args TEXT,
                    result TEXT,
                    error_message TEXT,
                    progress REAL DEFAULT 0.0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_status ON scheduled_tasks(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_type ON scheduled_tasks(task_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scheduled_time ON scheduled_tasks(scheduled_time DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_id ON scheduled_tasks(task_id)
            """)

            conn.commit()
            logger.info("Scheduler database tables initialized")

    def create_task(
        self,
        task_type: TaskType,
        scheduled_time: datetime,
        trigger_type: TriggerType = TriggerType.INTERVAL,
        trigger_args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledTask:
        """
        Create a new task record.

        Args:
            task_type: Type of task to create
            scheduled_time: When the task is scheduled to run
            trigger_type: Type of trigger (interval, manual, etc.)
            trigger_args: Additional trigger arguments
            metadata: Optional metadata for the task

        Returns:
            Created ScheduledTask
        """
        task = ScheduledTask.create(
            task_type=task_type,
            scheduled_time=scheduled_time,
            trigger_type=trigger_type,
            trigger_args=trigger_args,
            metadata=metadata,
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scheduled_tasks (
                    task_id, task_type, status, scheduled_time,
                    trigger_type, trigger_args, metadata, progress
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                task.task_type,
                task.status,
                task.scheduled_time.isoformat(),
                task.trigger_type,
                json.dumps(trigger_args) if trigger_args else None,
                json.dumps(metadata) if metadata else None,
                task.progress,
            ))
            conn.commit()

        logger.info(f"Created task {task.task_id} of type {task_type.value}")
        return task

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """
        Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            ScheduledTask or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM scheduled_tasks WHERE task_id = ?
            """, (task_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_task(row)
            return None

    def update_task_status(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        started_time: Optional[datetime] = None,
        completed_time: Optional[datetime] = None,
    ) -> bool:
        """
        Update task status and related fields.

        Args:
            task_id: Task identifier
            status: New status (optional)
            progress: Progress value 0.0-1.0 (optional)
            error_message: Error message if failed (optional)
            result: Result data (optional)
            started_time: Task start time (optional)
            completed_time: Task completion time (optional)

        Returns:
            True if updated successfully
        """
        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status.value)

        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        if result is not None:
            updates.append("result = ?")
            params.append(json.dumps(result))

        if started_time is not None:
            updates.append("started_time = ?")
            params.append(started_time.isoformat())

        if completed_time is not None:
            updates.append("completed_time = ?")
            params.append(completed_time.isoformat())

        # Always update updated_at
        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())

        if not updates:
            return False

        params.append(task_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE scheduled_tasks
                SET {', '.join(updates)}
                WHERE task_id = ?
            """, params)
            conn.commit()

            if cursor.rowcount > 0:
                logger.debug(f"Updated task {task_id}: {updates}")
                return True

        return False

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ScheduledTask]:
        """
        List tasks with optional filtering.

        Args:
            status: Filter by status (optional)
            task_type: Filter by task type (optional)
            limit: Maximum number of tasks to return
            offset: Number of tasks to skip

        Returns:
            List of ScheduledTasks
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build query
            conditions = []
            params = []

            if status:
                conditions.append("status = ?")
                params.append(status.value)

            if task_type:
                conditions.append("task_type = ?")
                params.append(task_type.value)

            query = "SELECT * FROM scheduled_tasks"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY scheduled_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

            return [self._row_to_task(row) for row in rows]

    def get_running_tasks(self) -> List[ScheduledTask]:
        """
        Get all currently running tasks.

        Returns:
            List of running ScheduledTasks
        """
        return self.list_tasks(status=TaskStatus.RUNNING, limit=100)

    def get_recent_tasks(self, limit: int = 10) -> List[ScheduledTask]:
        """
        Get most recent tasks.

        Args:
            limit: Number of tasks to return

        Returns:
            List of recent ScheduledTasks
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM scheduled_tasks
                ORDER BY scheduled_time DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [self._row_to_task(row) for row in rows]

    def cancel_task(self, task_id: str) -> bool:
        """
        Mark a task as cancelled (if not already completed).

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled successfully
        """
        task = self.get_task(task_id)
        if not task:
            return False

        if task.status in (TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value):
            logger.warning(f"Cannot cancel task {task_id} with status {task.status}")
            return False

        return self.update_task_status(
            task_id=task_id,
            status=TaskStatus.CANCELLED,
            completed_time=datetime.utcnow(),
        )

    def get_task_stats(self) -> Dict[str, int]:
        """
        Get task statistics.

        Returns:
            Dictionary with counts by status
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM scheduled_tasks")
            total = cursor.fetchone()[0]

            # Get counts by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM scheduled_tasks
                GROUP BY status
            """)

            stats = {"total": total}
            for row in cursor.fetchall():
                stats[row["status"]] = row["count"]

            return stats

    def cleanup_old_tasks(self, retention_days: int = 30) -> int:
        """
        Delete old completed/failed tasks beyond retention period.

        Args:
            retention_days: Number of days to retain task records

        Returns:
            Number of tasks deleted
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM scheduled_tasks
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND completed_time < ?
            """, (cutoff.isoformat(),))

            deleted = cursor.rowcount
            conn.commit()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old tasks (older than {retention_days} days)")

            return deleted

    def _row_to_task(self, row: sqlite3.Row) -> ScheduledTask:
        """Convert database row to ScheduledTask."""
        return ScheduledTask(
            task_id=row["task_id"],
            task_type=row["task_type"],
            status=row["status"],
            scheduled_time=datetime.fromisoformat(row["scheduled_time"]),
            started_time=datetime.fromisoformat(row["started_time"]) if row["started_time"] else None,
            completed_time=datetime.fromisoformat(row["completed_time"]) if row["completed_time"] else None,
            trigger_type=row["trigger_type"],
            trigger_args=json.loads(row["trigger_args"]) if row["trigger_args"] else None,
            result=json.loads(row["result"]) if row["result"] else None,
            error_message=row["error_message"],
            progress=row["progress"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )
