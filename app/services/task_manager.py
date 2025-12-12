"""Async task manager for background processing."""
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

from app.models import TranscriptionResponse, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents an async processing task."""
    id: str
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    result: Optional[TranscriptionResponse] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class TaskManager:
    """Manages async transcription tasks."""
    
    _instance: Optional["TaskManager"] = None
    _tasks: Dict[str, Task] = {}
    _semaphore: Optional[asyncio.Semaphore] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tasks = {}
        return cls._instance
    
    @classmethod
    def get_semaphore(cls, max_concurrent: int = 3) -> asyncio.Semaphore:
        """Get or create semaphore for limiting concurrent tasks."""
        if cls._semaphore is None:
            cls._semaphore = asyncio.Semaphore(max_concurrent)
        return cls._semaphore
    
    def create_task(self) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = Task(id=task_id)
        logger.info(f"Created task: {task_id}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)
    
    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        result: Optional[TranscriptionResponse] = None,
        error: Optional[str] = None,
    ):
        """Update task status."""
        task = self._tasks.get(task_id)
        if task:
            if status:
                task.status = status
            if progress is not None:
                task.progress = progress
            if result:
                task.result = result
                task.completed_at = datetime.now()
            if error:
                task.error = error
                task.completed_at = datetime.now()
    
    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status as response model."""
        task = self._tasks.get(task_id)
        if task:
            return TaskStatus(
                task_id=task.id,
                status=task.status,
                progress=task.progress,
                result=task.result,
                error=task.error,
            )
        return None
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Remove tasks older than max_age_hours."""
        now = datetime.now()
        to_remove = []
        
        for task_id, task in self._tasks.items():
            if task.completed_at:
                age = (now - task.completed_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self._tasks[task_id]
            logger.info(f"Cleaned up old task: {task_id}")


# Global task manager instance
task_manager = TaskManager()
