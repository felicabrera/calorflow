from .celery_app import celery
from .training import train_process_models
from .model_manager import save_model, save_checkpoint
from .db import SessionLocal, init_db, TrainingRun
import os
import redis
import os
import time
import json
from datetime import datetime
import logging
import time as _time
import os

REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
redis_client = redis.Redis.from_url(REDIS_URL)

try:
    import mlflow
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        try:
            import logging
            logging.getLogger(__name__).info('Celery MLflow tracking URI set to: %s', mlflow_tracking_uri)
        except Exception:
            pass
except Exception:
    pass


def publish_update(process: str, message: dict):
    channel = f"training:{process.upper()}"
    try:
        redis_client.publish(channel, json.dumps(message))
    except Exception:
        pass


class RedisLogHandler(logging.Handler):
    def __init__(self, redis_client, channel: str, process: str = 'UNKNOWN'):
        super().__init__()
        self.redis_client = redis_client
        self.channel = channel
        self.process = process

    def emit(self, record):
        try:
            msg = self.format(record)
            payload = {
                'event': 'log',
                'process': self.process,
                'ts': _time.time(),
                'level': record.levelname,
                'logger': record.name,
                'message': msg,
            }
            self.redis_client.publish(self.channel, json.dumps(payload))
        except Exception:
            pass


@celery.task(bind=True)
def train_process_task(self, process_name: str, n_trials: int = 20, use_optuna: bool = False):
    """Celery task wrapper around train_process_models."""
    process = process_name.upper()
    # Initialize DB
    init_db()
    db = SessionLocal()
    run = TrainingRun(process=process, status='running', start_ts=datetime.utcnow())
    db.add(run)
    db.commit()
    db.refresh(run)

    publish_update(process, {'event': 'started', 'ts': time.time(), 'message': 'Task started', 'progress': 0.01})
    logging.getLogger(__name__).info('Train process task starting %s', process)
    # attach Redis log handler so worker logs are pushed to UI via pubsub
    log_handler = RedisLogHandler(redis_client, f"training:{process}", process=process)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)
    logging.getLogger(__name__).info('Attached RedisLogHandler to root logger')
    try:
        train_df = None
        # Define a small closure to publish progress messages
        def publish_progress(msg: dict):
            # msg should include 'progress' and 'message' etc
            publish_update(process, msg)
            # update DB progress if available
            try:
                if 'progress' in msg:
                    db.query(TrainingRun).filter(TrainingRun.id == run.id).update({'status': 'running', 'progress': float(msg.get('progress', 0.0))})
                    db.commit()
            except Exception:
                pass

        # train_process_models handles reading processed files if needed
        result = train_process_models(train_df if train_df is not None else None, process, n_trials=n_trials, use_optuna=use_optuna, progress_callback=publish_progress)
        logging.getLogger(__name__).info('Training finished for %s; results metrics: %s', process, json.dumps({'pci': result['pci']['metrics'], 'h2': result['h2']['metrics']}))

        # save models
        for model_type, m in result['pci']['models'].items():
            path = save_model(process, 'PCI', model_type, m)
            # Ensure model file is present on shared volume before publishing save event
            for i in range(5):
                if os.path.exists(str(path)):
                    break
                _time.sleep(0.5)
            logging.getLogger(__name__).info('Saved PCI model %s for %s', model_type, process)
            publish_update(process, {'event': 'model_saved', 'path': str(path), 'model_type': model_type, 'target': 'PCI'})
        for model_type, m in result['h2']['models'].items():
            path = save_model(process, 'H2', model_type, m)
            # Ensure model file is present on shared volume before publishing save event
            for i in range(5):
                if os.path.exists(str(path)):
                    break
                _time.sleep(0.5)
            logging.getLogger(__name__).info('Saved H2 model %s for %s', model_type, process)
            publish_update(process, {'event': 'model_saved', 'path': str(path), 'model_type': model_type, 'target': 'H2'})

        metrics = {'pci': result['pci']['metrics'], 'h2': result['h2']['metrics']}
        save_checkpoint(process, 'results', {'metrics': metrics, 'feature_cols': result.get('feature_cols'), 'trained_at': time.time()})

        db.query(TrainingRun).filter(TrainingRun.id == run.id).update({'status': 'completed', 'end_ts': datetime.utcnow(), 'metrics': json.dumps(metrics), 'progress': 1.0})
        db.commit()
        publish_update(process, {'event': 'completed', 'ts': time.time(), 'metrics': metrics})
        return {'ok': True, 'metrics': metrics}
    except Exception as ex:
        db.query(TrainingRun).filter(TrainingRun.id == run.id).update({'status': 'error', 'end_ts': datetime.utcnow(), 'error': str(ex), 'progress': 0.0})
        db.commit()
        publish_update(process, {'event': 'error', 'ts': time.time(), 'message': str(ex)})
        raise
    finally:
        try:
            root_logger.removeHandler(log_handler)
        except Exception:
            pass


@celery.task(bind=True)
def train_sequence_task(self, processes: list, n_trials: int = 20, use_optuna: bool = False):
    """Train multiple processes sequentially (e.g., ['FCC','CCR']) in the same Celery task.
    Useful for preventing asynchronous metric interleaving when training BOTH sequentially.
    """
    # Initialize DB for each process run and call train_process_models sequentially
    init_db()
    db = SessionLocal()
    results = {}
    # attach Redis logger for sequence
    seq_channel = 'training:BOTH'
    seq_logger_handler = RedisLogHandler(redis_client, seq_channel, process='BOTH')
    root_logger = logging.getLogger()
    root_logger.addHandler(seq_logger_handler)
    root_logger.setLevel(logging.INFO)
    try:
        publish_update('BOTH', {'event': 'started', 'message': f'Sequential training starting: {processes}', 'progress': 0.01})
        logging.getLogger(__name__).info('Sequential training starting for %s', processes)
        total = len(processes)
        for idx, p in enumerate(processes):
            proc = p.upper()
            run = TrainingRun(process=proc, status='running', start_ts=datetime.utcnow())
            db.add(run)
            db.commit()
            db.refresh(run)
            publish_update(proc, {'event': 'started', 'message': f'Starting {proc}', 'progress': 0.01})
            logging.getLogger(__name__).info('Sequential task starting process %s', proc)

            def publish_progress(msg: dict):
                publish_update(proc, msg)
                try:
                    if 'progress' in msg:
                        db.query(TrainingRun).filter(TrainingRun.id == run.id).update({'status': 'running', 'progress': float(msg.get('progress', 0.0))})
                        db.commit()
                except Exception:
                    pass

            try:
                # call the internal training function directly
                # attach per-process log handler to stream logs to per-process channel
                proc_handler = RedisLogHandler(redis_client, f"training:{proc}", process=proc)
                root_logger.addHandler(proc_handler)
                res = train_process_models(None, proc, n_trials=n_trials, use_optuna=use_optuna, progress_callback=publish_progress)
                # Save models and checkpoint
                for model_type, m in res['pci']['models'].items():
                    path = save_model(proc, 'PCI', model_type, m)
                    logging.getLogger(__name__).info('Saved PCI model %s for %s', model_type, proc)
                    publish_update(proc, {'event': 'model_saved', 'path': str(path), 'model_type': model_type, 'target': 'PCI'})
                for model_type, m in res['h2']['models'].items():
                    path = save_model(proc, 'H2', model_type, m)
                    logging.getLogger(__name__).info('Saved H2 model %s for %s', model_type, proc)
                    publish_update(proc, {'event': 'model_saved', 'path': str(path), 'model_type': model_type, 'target': 'H2'})
                metrics = {'pci': res['pci']['metrics'], 'h2': res['h2']['metrics']}
                save_checkpoint(proc, 'results', {'metrics': metrics, 'feature_cols': res.get('feature_cols'), 'trained_at': time.time()})
                db.query(TrainingRun).filter(TrainingRun.id == run.id).update({'status': 'completed', 'end_ts': datetime.utcnow(), 'metrics': json.dumps(metrics), 'progress': 1.0})
                db.commit()
                publish_update(proc, {'event': 'completed', 'metrics': metrics, 'progress': 1.0})
                logging.getLogger(__name__).info('Process %s completed; metrics: %s', proc, json.dumps(metrics))
                results[proc] = metrics
            except Exception as ex:
                db.query(TrainingRun).filter(TrainingRun.id == run.id).update({'status': 'error', 'end_ts': datetime.utcnow(), 'error': str(ex), 'progress': 0.0})
                db.commit()
                publish_update(proc, {'event': 'error', 'message': str(ex)})
                raise
            finally:
                try:
                    root_logger.removeHandler(proc_handler)
                except Exception:
                    pass

            # publish overall sequence progress
            seq_progress = (idx + 1) / float(total)
            publish_update('BOTH', {'event': 'progress', 'progress': seq_progress})

        publish_update('BOTH', {'event': 'completed', 'metrics': results, 'progress': 1.0})
        return {'ok': True, 'metrics': results}
    except Exception as e:
        publish_update('BOTH', {'event': 'error', 'message': str(e)})
        raise
    finally:
        try:
            root_logger.removeHandler(seq_logger_handler)
        except Exception:
            pass
