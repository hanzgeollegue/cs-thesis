"""
Simple async processing system using threading and session-based progress tracking.
No external dependencies (no Celery needed).
"""
import threading
import time
import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Global dictionary to track running tasks
_TASKS = {}
_TASKS_LOCK = threading.Lock()


class TaskStatus:
    """Track status of an async batch processing task."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = 'pending'  # pending, processing, completed, failed
        self.progress = 0  # 0-100
        self.message = 'Initializing...'
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.total_resumes = 0
        self.processed_resumes = 0
    
    def update(self, status=None, progress=None, message=None, processed=None):
        """Update task status."""
        if status:
            self.status = status
        if progress is not None:
            self.progress = min(100, max(0, progress))
        if message:
            self.message = message
        if processed is not None:
            self.processed_resumes = processed
        self.updated_at = datetime.now()
    
    def to_dict(self):
        """Convert to dictionary for JSON response."""
        return {
            'task_id': self.task_id,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'result': self.result,
            'error': self.error,
            'total_resumes': self.total_resumes,
            'processed_resumes': self.processed_resumes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


def create_task(task_id: str, total_resumes: int) -> TaskStatus:
    """Create a new task and register it."""
    task = TaskStatus(task_id)
    task.total_resumes = total_resumes
    
    with _TASKS_LOCK:
        _TASKS[task_id] = task
    
    logger.info(f"Created async task: {task_id}")
    return task


def get_task(task_id: str) -> Optional[TaskStatus]:
    """Get task status by ID."""
    with _TASKS_LOCK:
        return _TASKS.get(task_id)


def update_task(task_id: str, **kwargs):
    """Update task status."""
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if task:
            task.update(**kwargs)


def cleanup_old_tasks(max_age_seconds: int = 3600):
    """Remove tasks older than max_age_seconds."""
    now = datetime.now()
    with _TASKS_LOCK:
        to_remove = []
        for task_id, task in _TASKS.items():
            age = (now - task.created_at).total_seconds()
            if age > max_age_seconds:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del _TASKS[task_id]
            logger.info(f"Cleaned up old task: {task_id}")


def process_batch_async(task_id: str, resume_paths, job_description, jd_criteria, disable_ocr):
    """Process batch asynchronously in background thread."""
    try:
        from .batch_processor import BatchProcessor
        from .config import get_llm_config
        
        logger.info(f"Starting async processing for task {task_id}")
        
        # Get task
        task = get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return
        
        # Update status
        update_task(task_id, status='processing', progress=10, message='Initializing batch processor...')
        
        # Get API key
        llm_cfg = get_llm_config()
        api_key = llm_cfg['api_key']
        
        # Initialize processor
        update_task(task_id, progress=15, message='Loading AI models...')
        processor = BatchProcessor(api_key=api_key, disable_ocr=disable_ocr)
        
        # Process batch with progress callbacks
        update_task(task_id, progress=20, message=f'Processing {len(resume_paths)} resumes...')
        
        result = processor.process_batch(
            resumes=resume_paths,
            job_description=job_description,
            jd_criteria=jd_criteria
        )
        
        # Check for errors
        if result and 'error' not in result:
            update_task(
                task_id,
                status='completed',
                progress=100,
                message='Processing complete!'
            )
            task.result = result
            logger.info(f"Task {task_id} completed successfully")
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
            update_task(
                task_id,
                status='failed',
                progress=0,
                message=f'Error: {error_msg}'
            )
            task.error = error_msg
            logger.error(f"Task {task_id} failed: {error_msg}")
        
    except Exception as e:
        logger.error(f"Error in async processing for task {task_id}: {e}", exc_info=True)
        update_task(
            task_id,
            status='failed',
            progress=0,
            message=f'Processing error: {str(e)}'
        )
        task = get_task(task_id)
        if task:
            task.error = str(e)


def start_batch_processing_async(resume_paths, job_description, jd_criteria, disable_ocr=False) -> str:
    """Start batch processing in background thread and return task ID."""
    # Generate unique task ID
    task_id = f"batch_{int(time.time())}_{os.urandom(4).hex()}"
    
    # Create task
    task = create_task(task_id, len(resume_paths))
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_batch_async,
        args=(task_id, resume_paths, job_description, jd_criteria, disable_ocr),
        daemon=True
    )
    thread.start()
    
    logger.info(f"Started async processing thread for task {task_id}")
    
    # Cleanup old tasks
    try:
        cleanup_old_tasks()
    except Exception as e:
        logger.warning(f"Error cleaning up old tasks: {e}")
    
    return task_id
