"""
IRIS Scheduler Module

Provides APScheduler-based task scheduling with persistent state management.
Replaces the simple while True + time.sleep() daemon implementation.

Components:
- TaskService: Database operations for task records
- SchedulerManager: APScheduler wrapper and management
- TaskExecutor: Task execution wrapper with progress tracking
- SchedulerOrchestrator: High-level orchestrator for update cycles
"""

from .task_models import TaskStatus, TaskType, TriggerType, ScheduledTask
from .task_service import TaskService
from .task_executor import TaskExecutor
from .scheduler_manager import SchedulerManager
from .scheduler_orchestrator import SchedulerOrchestrator

__all__ = [
    "TaskStatus",
    "TaskType",
    "TriggerType",
    "ScheduledTask",
    "TaskService",
    "TaskExecutor",
    "SchedulerManager",
    "SchedulerOrchestrator",
]
