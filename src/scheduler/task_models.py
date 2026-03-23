"""
Task data models for the IRIS scheduler.

Defines enums and data classes for task status tracking.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Dict
from uuid import uuid4


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Types of tasks that can be scheduled."""
    UPDATE_CYCLE = "update_cycle"
    INDEX_BUILD = "index_build"
    QA_PROCESSING = "qa_processing"


class TriggerType(str, Enum):
    """Types of task triggers."""
    INTERVAL = "interval"      # Fixed interval between runs
    CRON = "cron"              # Cron-like schedule
    MANUAL = "manual"          # Manually triggered
    ONE_TIME = "one_time"      # One-time scheduled task


@dataclass
class ScheduledTask:
    """Represents a scheduled task with full state tracking."""
    task_id: str
    task_type: str
    status: str
    scheduled_time: datetime
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    trigger_type: str = TriggerType.INTERVAL
    trigger_args: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        task_type: TaskType,
        scheduled_time: datetime,
        trigger_type: TriggerType = TriggerType.INTERVAL,
        trigger_args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ScheduledTask":
        """Create a new task with a generated UUID."""
        return cls(
            task_id=str(uuid4()),
            task_type=task_type.value,
            status=TaskStatus.PENDING.value,
            scheduled_time=scheduled_time,
            trigger_type=trigger_type.value,
            trigger_args=trigger_args,
            metadata=metadata,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif value is None and key.endswith("_time"):
                data[key] = None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledTask":
        """Create from dictionary (deserialization)."""
        # Parse ISO format datetime strings
        for key in ["scheduled_time", "started_time", "completed_time", "created_at", "updated_at"]:
            if data.get(key) and isinstance(data[key], str):
                try:
                    data[key] = datetime.fromisoformat(data[key].replace("Z", "+00:00"))
                except ValueError:
                    pass  # Keep as string if parsing fails

        # Parse JSON strings for complex fields
        if isinstance(data.get("trigger_args"), str):
            try:
                data["trigger_args"] = json.loads(data["trigger_args"])
            except (json.JSONDecodeError, TypeError):
                data["trigger_args"] = None

        if isinstance(data.get("result"), str):
            try:
                data["result"] = json.loads(data["result"])
            except (json.JSONDecodeError, TypeError):
                data["result"] = None

        if isinstance(data.get("metadata"), str):
            try:
                data["metadata"] = json.loads(data["metadata"])
            except (json.JSONDecodeError, TypeError):
                data["metadata"] = None

        return cls(**data)


@dataclass
class JobInfo:
    """Information about a scheduled job."""
    job_id: str
    name: str
    next_run_time: Optional[str]
    trigger: str
