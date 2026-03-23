"""
Scheduler Orchestrator for IRIS

High-level orchestrator that manages scheduled update cycles using APScheduler.
Replaces the simple DaemonOrchestrator with persistent state management.
"""

import logging
from typing import Any, Dict, List, Optional

from src.config import get_config
from core.orchestrator import UpdateOrchestrator

from .task_service import TaskService
from .scheduler_manager import SchedulerManager
from .task_models import ScheduledTask, TaskType, JobInfo

logger = logging.getLogger(__name__)


class SchedulerOrchestrator:
    """
    High-level orchestrator that manages scheduled update cycles.

    Replaces DaemonOrchestrator with APScheduler-based implementation:
    - Persistent task state management
    - Task status tracking
    - Job scheduling and management
    - Query and control capabilities

    Usage:
        orchestrator = SchedulerOrchestrator(config)
        orchestrator.start_scheduler(interval_hours=24)
        # ... scheduler runs in background ...
        orchestrator.stop_scheduler()

        # Query task status
        tasks = orchestrator.list_recent_tasks()
        status = orchestrator.get_task_status(task_id)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize scheduler orchestrator.

        Args:
            config: Configuration dictionary (uses default if None)
        """
        self.config = config or get_config().config

        # Initialize services
        self.task_service = TaskService(self.config["storage"]["paper_db_path"])
        self.scheduler_manager = SchedulerManager(self.config, self.task_service)

        # Create base orchestrator for actual update logic
        self.base_orchestrator = UpdateOrchestrator(self.config)

        # Set orchestrator reference for scheduler manager
        self.scheduler_manager.set_orchestrator(self.base_orchestrator)

        logger.info("Scheduler orchestrator initialized")

    def start_scheduler(self, interval_hours: Optional[int] = None, run_immediately: bool = True) -> None:
        """
        Start the scheduler with periodic update cycles.

        Args:
            interval_hours: Hours between cycles (default: from config)
            run_immediately: Whether to run first task immediately on startup (default: True)
        """
        scheduler_config = self.config.get("scheduler", {})
        interval = interval_hours or scheduler_config.get(
            "default_interval_hours",
            self.config.get("update", {}).get("interval_hours", 24)
        )

        # Schedule periodic update cycles
        self.scheduler_manager.schedule_interval_task(
            task_type=TaskType.UPDATE_CYCLE,
            interval_hours=interval,
            task_func=self._run_update_cycle,
            job_id="periodic_update_cycle",
        )

        # Start scheduler
        self.scheduler_manager.start()

        # Run first task immediately on startup
        if run_immediately:
            logger.info("Running first update cycle immediately on startup...")
            self.run_single_cycle()

        logger.info(f"Scheduler started with {interval}h interval")
        logger.info("Press Ctrl+C to stop")

    def stop_scheduler(self) -> None:
        """Stop the scheduler."""
        self.scheduler_manager.shutdown()

    def run_single_cycle(self) -> bool:
        """
        Run a single update cycle immediately.

        Returns:
            True if cycle triggered successfully
        """
        try:
            job_id = self.scheduler_manager.run_task_now(
                task_type=TaskType.UPDATE_CYCLE,
                task_func=self._run_update_cycle,
            )
            logger.info(f"Single update cycle triggered: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to trigger update cycle: {e}")
            return False

    def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """
        Get status of a specific task.

        Args:
            task_id: Task identifier

        Returns:
            ScheduledTask or None if not found
        """
        return self.task_service.get_task(task_id)

    def list_recent_tasks(self, limit: int = 10) -> List[ScheduledTask]:
        """
        Get recent tasks across all types.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of ScheduledTasks
        """
        return self.task_service.get_recent_tasks(limit=limit)

    def list_running_tasks(self) -> List[ScheduledTask]:
        """
        Get all currently running tasks.

        Returns:
            List of running ScheduledTasks
        """
        return self.task_service.get_running_tasks()

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running or pending task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled successfully
        """
        return self.task_service.cancel_task(task_id)

    def get_scheduled_jobs(self) -> List[JobInfo]:
        """
        Get information about scheduled jobs.

        Returns:
            List of JobInfo objects
        """
        return self.scheduler_manager.get_scheduled_jobs()

    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get overall scheduler status.

        Returns:
            Dictionary with scheduler status information
        """
        stats = self.task_service.get_task_stats()
        jobs = self.scheduler_manager.get_scheduled_jobs()

        return {
            "running": self.scheduler_manager.is_running(),
            "task_stats": stats,
            "scheduled_jobs": len(jobs),
            "jobs": [j.to_dict() if hasattr(j, 'to_dict') else {
                "job_id": j.job_id,
                "name": j.name,
                "next_run_time": j.next_run_time,
                "trigger": j.trigger,
            } for j in jobs],
        }

    def cleanup_old_tasks(self, retention_days: Optional[int] = None) -> int:
        """
        Clean up old task records.

        Args:
            retention_days: Days to retain (default: from config)

        Returns:
            Number of tasks deleted
        """
        scheduler_config = self.config.get("scheduler", {})
        retention = retention_days or scheduler_config.get("history_retention_days", 30)

        return self.task_service.cleanup_old_tasks(retention_days=retention)

    def _run_update_cycle(self) -> bool:
        """
        Run a single update cycle (called by scheduler).

        Delegates to the base UpdateOrchestrator.

        Returns:
            True if cycle completed successfully
        """
        return self.base_orchestrator.run_cycle()
