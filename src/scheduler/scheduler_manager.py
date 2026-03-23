"""
Scheduler Manager for IRIS

Manages APScheduler instance for periodic task scheduling.
Provides job scheduling, cancellation, and status query capabilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.executors.pool import ThreadPoolExecutor
except ImportError:
    raise ImportError(
        "APScheduler is required. Install it with: pip install apscheduler>=3.10.0"
    )

from .task_service import TaskService
from .task_executor import TaskExecutor
from .task_models import TaskType, TriggerType, JobInfo

logger = logging.getLogger(__name__)


class SchedulerManager:
    """
    Manages APScheduler instance for periodic task scheduling.

    Features:
    - Interval-based scheduling (fixed hours between runs)
    - One-time scheduled tasks
    - Manual task triggering
    - Job persistence (via database-backed job store)
    - Concurrent execution prevention
    - Graceful shutdown
    """

    def __init__(self, config: Dict[str, Any], task_service: TaskService):
        """
        Initialize scheduler manager.

        Args:
            config: Configuration dictionary
            task_service: Task database service
        """
        self.config = config
        self.task_service = task_service
        self.executor = TaskExecutor(task_service)

        # Configure job stores
        jobstores = {
            'default': MemoryJobStore(),  # Simple in-memory store
        }

        # Configure executor
        executors = {
            'default': ThreadPoolExecutor(max_workers=1),  # Sequential execution
        }

        # Scheduler configuration
        scheduler_config = config.get("scheduler", {})
        self.timezone = scheduler_config.get("timezone", "UTC")

        # Create scheduler
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            timezone=self.timezone,
            job_defaults={
                'coalesce': True,  # Combine missed jobs into one
                'max_instances': 1,  # Prevent overlapping executions
                'misfire_grace_time': 3600,  # Grace period for missed jobs
            }
        )

        # Keep track of orchestrator instance for job execution
        self.orchestrator: Optional[Any] = None

        logger.info("Scheduler manager initialized")

    def set_orchestrator(self, orchestrator: Any):
        """
        Set the orchestrator instance for job execution.

        Args:
            orchestrator: Orchestrator instance with run_cycle() method
        """
        self.orchestrator = orchestrator

    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")
        else:
            logger.warning("Scheduler already running")

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the scheduler.

        Args:
            wait: Wait for running jobs to complete
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("Scheduler shutdown")

    def schedule_interval_task(
        self,
        task_type: TaskType,
        interval_hours: int,
        task_func: Callable,
        start_date: Optional[datetime] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Schedule a task to run at fixed intervals.

        Args:
            task_type: Type of task to schedule
            interval_hours: Hours between executions
            task_func: Function to execute
            start_date: When to start scheduling (default: now)
            job_id: Unique job identifier (default: auto-generated)

        Returns:
            Job ID for the scheduled task
        """
        if job_id is None:
            job_id = f"{task_type.value}_interval"

        self.scheduler.add_job(
            func=self._wrap_task_execution(task_type, task_func),
            trigger=IntervalTrigger(hours=interval_hours, start_date=start_date),
            id=job_id,
            name=f"{task_type.value} (interval: {interval_hours}h)",
            replace_existing=True,
        )

        logger.info(f"Scheduled interval task: {job_id} (interval: {interval_hours}h)")
        return job_id

    def schedule_one_time_task(
        self,
        task_type: TaskType,
        run_date: datetime,
        task_func: Callable,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Schedule a one-time task to run at specific date/time.

        Args:
            task_type: Type of task to schedule
            run_date: When to run the task
            task_func: Function to execute
            job_id: Unique job identifier (default: auto-generated)

        Returns:
            Job ID
        """
        if job_id is None:
            job_id = f"{task_type.value}_onetime_{run_date.timestamp()}"

        self.scheduler.add_job(
            func=self._wrap_task_execution(task_type, task_func),
            trigger=DateTrigger(run_date=run_date),
            id=job_id,
            name=f"{task_type.value} at {run_date}",
            replace_existing=False,
        )

        logger.info(f"Scheduled one-time task: {job_id} at {run_date}")
        return job_id

    def run_task_now(self, task_type: TaskType, task_func: Callable) -> str:
        """
        Trigger a task to run immediately.

        Args:
            task_type: Type of task to run
            task_func: Function to execute

        Returns:
            Job ID
        """
        return self.schedule_one_time_task(
            task_type=task_type,
            run_date=datetime.now(),
            task_func=task_func,
        )

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a scheduled job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled successfully
        """
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Cancelled job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def pause_job(self, job_id: str) -> bool:
        """
        Pause a scheduled job.

        Args:
            job_id: Job identifier

        Returns:
            True if paused successfully
        """
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        """
        Resume a paused job.

        Args:
            job_id: Job identifier

        Returns:
            True if resumed successfully
        """
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False

    def get_scheduled_jobs(self) -> List[JobInfo]:
        """
        Get information about all scheduled jobs.

        Returns:
            List of JobInfo objects
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append(JobInfo(
                job_id=job.id,
                name=job.name,
                next_run_time=job.next_run_time.isoformat() if job.next_run_time else None,
                trigger=str(job.trigger),
            ))
        return jobs

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """
        Get information about a specific job.

        Args:
            job_id: Job identifier

        Returns:
            JobInfo or None if not found
        """
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                return JobInfo(
                    job_id=job.id,
                    name=job.name,
                    next_run_time=job.next_run_time.isoformat() if job.next_run_time else None,
                    trigger=str(job.trigger),
                )
        except Exception:
            pass
        return None

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self.scheduler.running

    def _wrap_task_execution(self, task_type: TaskType, task_func: Callable) -> Callable:
        """
        Wrap task function to create database record and track execution.

        Returns a callable suitable for APScheduler.

        Args:
            task_type: Type of task
            task_func: Function to execute

        Returns:
            Wrapped callable
        """
        def wrapped():
            # Create task record
            task = self.task_service.create_task(
                task_type=task_type,
                scheduled_time=datetime.utcnow(),
                trigger_type=TriggerType.INTERVAL,
                metadata={"job_id": "scheduled"},
            )

            # Execute with tracking
            try:
                result = self.executor.execute_task(task.task_id, task_func)
                return result
            except Exception as e:
                logger.error(f"Task execution error: {e}")
                raise

        return wrapped
