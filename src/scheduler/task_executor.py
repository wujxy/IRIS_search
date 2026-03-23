"""
Task Executor for IRIS Scheduler

Wraps task execution with comprehensive status tracking and progress reporting.
"""

import asyncio
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Optional, Dict

from .task_service import TaskService
from .task_models import ScheduledTask, TaskStatus

logger = logging.getLogger(__name__)


class TaskExecutionError(Exception):
    """Raised when task execution fails."""
    pass


class TaskExecutor:
    """
    Wraps task function execution with status tracking and logging.

    This class provides:
    - Automatic status updates (pending -> running -> completed/failed)
    - Progress reporting
    - Error capturing and logging
    - Support for both sync and async task functions
    """

    def __init__(self, task_service: TaskService):
        """
        Initialize task executor.

        Args:
            task_service: Task database service for status updates
        """
        self.task_service = task_service

    def execute_task(
        self,
        task_id: str,
        task_func: Callable,
        **kwargs
    ) -> Any:
        """
        Execute a task function with comprehensive status tracking.

        Handles both synchronous and asynchronous task functions.

        Args:
            task_id: Task identifier
            task_func: Function to execute (sync or async)
            **kwargs: Arguments to pass to task_func

        Returns:
            Result from task_func

        Raises:
            TaskExecutionError: If task execution fails
        """
        # Update status to RUNNING
        self.task_service.update_task_status(
            task_id,
            status=TaskStatus.RUNNING,
            started_time=datetime.utcnow(),
        )

        try:
            # Execute task (handle both sync and async)
            if asyncio.iscoroutinefunction(task_func):
                # Run async function in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(task_func(**kwargs))
                finally:
                    loop.close()
            else:
                result = task_func(**kwargs)

            # Update status to COMPLETED
            self.task_service.update_task_status(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=1.0,
                completed_time=datetime.utcnow(),
                result={"success": True, "data": self._serialize_result(result)},
            )

            logger.info(f"Task {task_id} completed successfully")
            return result

        except Exception as e:
            # Update status to FAILED
            error_msg = f"{type(e).__name__}: {str(e)}"
            error_traceback = traceback.format_exc()

            self.task_service.update_task_status(
                task_id,
                status=TaskStatus.FAILED,
                completed_time=datetime.utcnow(),
                error_message=error_msg,
                result={"success": False, "error": error_msg},
            )

            logger.error(f"Task {task_id} failed: {error_msg}\n{error_traceback}")
            raise TaskExecutionError(f"Task execution failed: {error_msg}") from e

    def report_progress(self, task_id: str, progress: float, message: Optional[str] = None):
        """
        Update task progress.

        Args:
            task_id: Task identifier
            progress: Progress value (0.0 to 1.0)
            message: Optional progress message
        """
        if not 0.0 <= progress <= 1.0:
            logger.warning(f"Invalid progress value: {progress}, must be between 0.0 and 1.0")
            progress = max(0.0, min(1.0, progress))

        self.task_service.update_task_status(task_id, progress=progress)

        if message:
            logger.debug(f"Task {task_id} progress: {progress:.1%} - {message}")

    async def execute_task_async(
        self,
        task_id: str,
        task_func: Callable,
        **kwargs
    ) -> Any:
        """
        Execute a task function asynchronously with status tracking.

        Args:
            task_id: Task identifier
            task_func: Async function to execute
            **kwargs: Arguments to pass to task_func

        Returns:
            Result from task_func

        Raises:
            TaskExecutionError: If task execution fails
        """
        # Update status to RUNNING
        self.task_service.update_task_status(
            task_id,
            status=TaskStatus.RUNNING,
            started_time=datetime.utcnow(),
        )

        try:
            # Execute async task
            result = await task_func(**kwargs)

            # Update status to COMPLETED
            self.task_service.update_task_status(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=1.0,
                completed_time=datetime.utcnow(),
                result={"success": True, "data": self._serialize_result(result)},
            )

            logger.info(f"Task {task_id} completed successfully")
            return result

        except Exception as e:
            # Update status to FAILED
            error_msg = f"{type(e).__name__}: {str(e)}"
            error_traceback = traceback.format_exc()

            self.task_service.update_task_status(
                task_id,
                status=TaskStatus.FAILED,
                completed_time=datetime.utcnow(),
                error_message=error_msg,
                result={"success": False, "error": error_msg},
            )

            logger.error(f"Task {task_id} failed: {error_msg}\n{error_traceback}")
            raise TaskExecutionError(f"Task execution failed: {error_msg}") from e

    def _serialize_result(self, result: Any) -> Any:
        """
        Serialize task result for JSON storage.

        Args:
            result: Result to serialize

        Returns:
            JSON-serializable result
        """
        if result is None:
            return None
        elif isinstance(result, (bool, int, float, str)):
            return result
        elif isinstance(result, (list, tuple)):
            return str(result)  # Convert to string representation
        elif isinstance(result, dict):
            return {k: str(v) for k, v in result.items()}
        else:
            return str(result)


class ProgressReporter:
    """
    Context manager for reporting task progress.

    Usage:
        with ProgressReporter(executor, task_id) as reporter:
            reporter.update(0.3, "Step 1 complete")
            # ... do work ...
            reporter.update(0.7, "Step 2 complete")
    """

    def __init__(self, executor: TaskExecutor, task_id: str):
        """
        Initialize progress reporter.

        Args:
            executor: TaskExecutor instance
            task_id: Task identifier
        """
        self.executor = executor
        self.task_id = task_id

    def update(self, progress: float, message: Optional[str] = None):
        """Update progress."""
        self.executor.report_progress(self.task_id, progress, message)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type is None:
            self.update(1.0, "Complete")
        return False
