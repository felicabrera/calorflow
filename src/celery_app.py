from celery import Celery
import os

# Broker and backend are provided via env variables; fallback to Redis local
REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
BACKEND = os.environ.get('CELERY_RESULT_BACKEND', REDIS_URL)

celery = Celery('calorflow_tasks', broker=REDIS_URL, backend=BACKEND)

# Basic config (can be expanded)
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Ensure task modules are imported so the worker registers tasks like src.tasks.train_process_task
try:
    # Prefer absolute import when available
    import src.tasks  # noqa: F401
except Exception:
    try:
        from . import tasks  # noqa: F401
    except Exception:
        pass

# Try to help celery discover tasks
try:
    celery.autodiscover_tasks(['src.tasks'])
except Exception:
    pass

# Log registered tasks that belong to our package for easier debugging.
try:
    import logging
    logger = logging.getLogger(__name__)
    registered = [t for t in celery.tasks.keys() if t.startswith('src.tasks.')]
    logger.info('Detected registered celery tasks: %s', registered)
except Exception:
    pass

if __name__ == '__main__':
    print('Celery app config: ', celery.conf)
