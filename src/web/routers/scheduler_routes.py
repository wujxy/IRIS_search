"""
Scheduler API routes for IRIS Web module.

Provides endpoints for task scheduling and management.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

from scheduler.task_service import TaskService
from scheduler.scheduler_orchestrator import SchedulerOrchestrator
from scheduler.task_models import TaskStatus, TaskType

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


# Request/Response Models
class SchedulerStatusResponse(BaseModel):
    """Scheduler status response."""
    running: bool
    task_stats: dict
    scheduled_jobs: int
    jobs: List[dict]


class TaskResponse(BaseModel):
    """Task response."""
    task_id: str
    task_type: str
    status: str
    scheduled_time: str
    started_time: Optional[str] = None
    completed_time: Optional[str] = None
    trigger_type: str
    error_message: Optional[str] = None
    progress: float
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TaskListResponse(BaseModel):
    """Task list response."""
    tasks: List[TaskResponse]
    total: int


class CancelResponse(BaseModel):
    """Cancel task response."""
    success: bool
    message: str


class RunNowResponse(BaseModel):
    """Run now response."""
    success: bool
    job_id: Optional[str] = None
    message: str


# Dependency for scheduler orchestrator
# Note: This is a simple singleton for now. In production, use proper dependency injection.
_scheduler_orchestrator: Optional[SchedulerOrchestrator] = None


def set_scheduler_orchestrator(orchestrator: SchedulerOrchestrator):
    """Set the global scheduler orchestrator instance."""
    global _scheduler_orchestrator
    _scheduler_orchestrator = orchestrator


def get_scheduler_orchestrator() -> SchedulerOrchestrator:
    """Get the global scheduler orchestrator instance."""
    if _scheduler_orchestrator is None:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    return _scheduler_orchestrator


@router.get("/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status():
    """
    Get scheduler status and upcoming jobs.

    Returns information about:
    - Whether scheduler is running
    - Task statistics (total, by status)
    - Scheduled jobs count
    - Job details
    """
    orchestrator = get_scheduler_orchestrator()
    status = orchestrator.get_scheduler_status()

    return SchedulerStatusResponse(**status)


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List tasks with optional filtering.

    - **status**: Filter by status (pending, running, completed, failed, cancelled)
    - **task_type**: Filter by task type (update_cycle, index_build, qa_processing)
    - **limit**: Maximum number of tasks to return
    - **offset**: Number of tasks to skip
    """
    orchestrator = get_scheduler_orchestrator()
    task_service = orchestrator.task_service

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = TaskStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    # Parse task_type filter
    type_filter = None
    if task_type:
        try:
            type_filter = TaskType(task_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid task_type: {task_type}")

    tasks = task_service.list_tasks(
        status=status_filter,
        task_type=type_filter,
        limit=limit,
        offset=offset,
    )

    # Get total count (for pagination info, would need additional query)
    task_responses = [TaskResponse(**task.to_dict()) for task in tasks]

    return TaskListResponse(
        tasks=task_responses,
        total=len(task_responses),
    )


@router.get("/tasks/recent", response_model=TaskListResponse)
async def get_recent_tasks(
    limit: int = Query(10, ge=1, le=100),
):
    """
    Get recent tasks across all types.

    - **limit**: Maximum number of tasks to return
    """
    orchestrator = get_scheduler_orchestrator()
    tasks = orchestrator.list_recent_tasks(limit=limit)

    task_responses = [TaskResponse(**task.to_dict()) for task in tasks]

    return TaskListResponse(
        tasks=task_responses,
        total=len(task_responses),
    )


@router.get("/tasks/running", response_model=TaskListResponse)
async def get_running_tasks():
    """Get all currently running tasks."""
    orchestrator = get_scheduler_orchestrator()
    tasks = orchestrator.list_running_tasks()

    task_responses = [TaskResponse(**task.to_dict()) for task in tasks]

    return TaskListResponse(
        tasks=task_responses,
        total=len(task_responses),
    )


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """
    Get details of a specific task.

    - **task_id**: Task identifier (UUID)
    """
    orchestrator = get_scheduler_orchestrator()
    task = orchestrator.get_task_status(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return TaskResponse(**task.to_dict())


@router.post("/tasks/{task_id}/cancel", response_model=CancelResponse)
async def cancel_task(task_id: str):
    """
    Cancel a running or pending task.

    - **task_id**: Task identifier (UUID)
    """
    orchestrator = get_scheduler_orchestrator()
    task = orchestrator.get_task_status(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.status in (TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value):
        return CancelResponse(
            success=False,
            message=f"Cannot cancel task with status {task.status}",
        )

    success = orchestrator.cancel_task(task_id)

    if success:
        return CancelResponse(success=True, message=f"Task {task_id} cancelled")
    else:
        return CancelResponse(success=False, message=f"Failed to cancel task {task_id}")


@router.post("/tasks/run-now", response_model=RunNowResponse)
async def run_task_now():
    """
    Trigger an immediate update cycle.

    Creates a one-time task that runs immediately.
    """
    orchestrator = get_scheduler_orchestrator()

    if not orchestrator.scheduler_manager.is_running():
        return RunNowResponse(
            success=False,
            job_id=None,
            message="Scheduler is not running. Start the scheduler first.",
        )

    success = orchestrator.run_single_cycle()

    if success:
        # Get the most recent task (the one we just created)
        recent = orchestrator.list_recent_tasks(limit=1)
        job_id = recent[0].task_id if recent else None

        return RunNowResponse(
            success=True,
            job_id=job_id,
            message="Update cycle triggered successfully",
        )
    else:
        return RunNowResponse(
            success=False,
            job_id=None,
            message="Failed to trigger update cycle",
        )


@router.get("/jobs")
async def get_scheduled_jobs():
    """Get all scheduled jobs."""
    orchestrator = get_scheduler_orchestrator()
    jobs = orchestrator.get_scheduled_jobs()

    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "name": job.name,
                "next_run_time": job.next_run_time,
                "trigger": job.trigger,
            }
            for job in jobs
        ]
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for scheduler."""
    try:
        orchestrator = get_scheduler_orchestrator()
        status = orchestrator.get_scheduler_status()

        return {
            "status": "healthy" if status["running"] else "stopped",
            "running": status["running"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Scheduler unhealthy: {str(e)}")
