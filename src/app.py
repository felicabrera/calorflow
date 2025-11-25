from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, Body
import os
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import asyncio
import time
from typing import Dict, Any
import json

from .schemas import TrainRequest, PredictRequest, TrainStatus
from pydantic import BaseModel
from typing import Optional
from .data_processing import prepare_fcc_data, prepare_ccr_data, add_feature_defaults
from .training import train_process_models, predict_with_ensemble
from .model_manager import save_model, load_model, save_checkpoint, load_checkpoint, list_models
from .model_manager import delete_model_by_path, delete_models_for_process, delete_all_models
from .tasks import train_process_task, publish_update, train_sequence_task
from .db import init_db, SessionLocal, TrainingRun

app = FastAPI(title='CalorFlow 2 API', version='0.1.0')

# Allow CORS for the frontend (webpack/vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

DATA_ROOT = Path('.')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_JOBS: Dict[str, Any] = {}
TRAIN_LOGS: Dict[str, list] = {}


def append_log(process: str, message: str):
    p = process.upper()
    if p not in TRAIN_LOGS:
        TRAIN_LOGS[p] = []
    if len(TRAIN_LOGS[p]) == 0 or TRAIN_LOGS[p][-1] != message:
        TRAIN_LOGS[p].append(message)


@app.on_event('startup')
def startup_event():
    try:
        init_db()
    except Exception as e:
        print('DB init failed:', e)

# Serve static files for the web UI
static_dir = Path('web') / 'static'
if static_dir.exists():
    app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')


@app.get('/')
def index():
    file = Path('web/index.html')
    if file.exists():
        return FileResponse(file)
    return HTMLResponse('<h2>CalorFlow API Server</h2><p>Visit /docs for Swagger UI</p>')


@app.get('/api/health')
def health():
    import platform
    return {
        'status': 'ok',
        'platform': platform.system(),
        'python_version': platform.python_version(),
    }


@app.post('/api/preprocess')
def preprocess(force: bool = False):
    """Run preprocessing step and produce train/test csvs (non-blocking for now)."""
    parent_dir = Path('.')
    processed_dir = parent_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Run FCC and CCR prepare - these write processed files
    try:
        fcc_train, fcc_test = prepare_fcc_data(str(parent_dir / 'data' / 'FCC - Cracking Catalítico'))
        ccr_train, ccr_test = prepare_ccr_data(str(parent_dir / 'data' / 'CCR - Reforming Catalítico'))

        fcc_train.to_csv(processed_dir / 'fcc_train.csv', index=False)
        fcc_test.to_csv(processed_dir / 'fcc_test.csv', index=False)
        ccr_train.to_csv(processed_dir / 'ccr_train.csv', index=False)
        ccr_test.to_csv(processed_dir / 'ccr_test.csv', index=False)

        return {'ok': True, 'message': 'Preprocessing complete', 'files': [str(p) for p in processed_dir.glob('*.csv')]}
    except Exception as e:
        # Detailed error for easier debugging
        import traceback
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail={'error': str(e), 'traceback': traceback_str})


@app.post('/api/train')
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    process = request.process.upper()
    if process not in ['FCC', 'CCR', 'BOTH']:
        raise HTTPException(status_code=400, detail="Invalid process - choose 'FCC', 'CCR' or 'BOTH'")

    # If the user requested BOTH, ensure neither process is already running
    if process == 'BOTH':
        if TRAIN_JOBS.get('FCC', {}).get('running', False) or TRAIN_JOBS.get('CCR', {}).get('running', False):
            raise HTTPException(status_code=409, detail='Training already running for FCC or CCR')
    else:
        if TRAIN_JOBS.get(process, {}).get('running', False):
            raise HTTPException(status_code=409, detail='Training already running for this process')

    # Create status (for BOTH we create per-process entries below)
    if process != 'BOTH':
        TRAIN_JOBS[process] = {
            'running': True,
            'progress': 0.0,
            'start_ts': time.time(),
            'last_update': time.time(),
            'metrics': {}
        }
        TRAIN_LOGS[process] = []

    def _run_training(process_name: str, n_trials: int, use_optuna: bool):
        try:
            # Load processed train
            processed_dir = Path('.') / 'data' / 'processed'
            processed_file = processed_dir / (process_name.lower() + '_train.csv')
            if not processed_file.exists():
                # Run preprocessing automatically
                from .data_processing import prepare_fcc_data, prepare_ccr_data
                if process_name == 'FCC':
                    train_df, _ = prepare_fcc_data(str(Path('.') / 'data' / 'FCC - Cracking Catalítico'))
                else:
                    train_df, _ = prepare_ccr_data(str(Path('.') / 'data' / 'CCR - Reforming Catalítico'))
            else:
                train_df = pd.read_csv(processed_file)

            # Update progress - use process_name local variable
            TRAIN_JOBS[process_name]['progress'] = 0.05
            TRAIN_JOBS[process_name]['last_update'] = time.time()
            # Append logs with dedup
            def append_log(p, m):
                if p not in TRAIN_LOGS:
                    TRAIN_LOGS[p] = []
                if len(TRAIN_LOGS[p]) == 0 or TRAIN_LOGS[p][-1] != m:
                    TRAIN_LOGS[p].append(m)

            append_log(process_name, 'Data loaded, starting training...')

            # Train models
            # Run via Celery if available
            # try:
            #     # schedule celery task and return job id
            #     task = train_process_task.delay(process_name, n_trials, use_optuna)
            #     TRAIN_JOBS[process_name]['celery_task_id'] = task.id
            #     # Store DB entry for this task
            #     try:
            #         db = SessionLocal()
            #         tr = TrainingRun(process=process_name, status='queued')
            #         db.add(tr)
            #         db.commit()
            #     except Exception:
            #         pass
            #     append_log(process_name, 'Training scheduled (celery) - task id: ' + str(task.id))
            #     # We won't block here - job will run asynchronously via Celery
            #     return
            # except Exception:
            #     # fallback local training
            publish_update(process_name, {'event': 'local_training_started', 'message': 'Local training fallback active'})
            result = train_process_models(train_df, process_name, n_trials=n_trials, use_optuna=use_optuna, progress_callback=lambda m: publish_update(process_name, m))
            publish_update(process_name, {'event': 'local_training_completed', 'metrics': {'pci': result['pci']['metrics'], 'h2': result['h2']['metrics']}})
            # Save models
            for model_type, m in result['pci']['models'].items():
                save_model(process_name, 'PCI', model_type, m)
            for model_type, m in result['h2']['models'].items():
                save_model(process_name, 'H2', model_type, m)

            TRAIN_JOBS[process_name]['progress'] = 0.95
            append_log(process_name, 'Models trained, saving...')
            TRAIN_JOBS[process_name]['metrics'] = {
                'pci': result['pci']['metrics'],
                'h2': result['h2']['metrics']
            }

            TRAIN_JOBS[process_name]['running'] = False
            append_log(process_name, 'Done')
            TRAIN_JOBS[process_name]['progress'] = 1.0
            TRAIN_JOBS[process_name]['last_update'] = time.time()

            save_checkpoint(process_name, 'results', {
                'metrics': TRAIN_JOBS[process]['metrics'],
                'feature_cols': result.get('feature_cols'),
                'imputation_values': result.get('imputation_values'),
                'trained_at': time.time()
            })
        except Exception as e:
            TRAIN_JOBS[process_name]['running'] = False
            TRAIN_JOBS[process_name]['progress'] = 0.0
            TRAIN_JOBS[process_name]['last_update'] = time.time()
            TRAIN_JOBS[process_name]['error'] = str(e)
            append_log(process_name, 'ERROR: ' + str(e))

    def _run_training_sequence(n_trials: int, use_optuna: bool):
        # Run FCC then CCR sequentially
        _run_training('FCC', n_trials, use_optuna)
        _run_training('CCR', n_trials, use_optuna)

    # If user asked to train both processes at once
    if process == 'BOTH':
        # set both running
        TRAIN_JOBS['FCC'] = {'running': True, 'progress': 0.0, 'start_ts': time.time(), 'metrics': {}}
        TRAIN_LOGS['FCC'] = []
        TRAIN_JOBS['CCR'] = {'running': True, 'progress': 0.0, 'start_ts': time.time(), 'metrics': {}}
        TRAIN_LOGS['CCR'] = []
        # prefer celery sequence task if available
        try:
            task = train_sequence_task.delay(['FCC', 'CCR'], request.n_trials, request.use_optuna)
            TRAIN_JOBS['BOTH'] = {'running': True, 'celery_task_id': task.id, 'progress': 0.01}
            TRAIN_LOGS['BOTH'] = [f'Training scheduled (sequential celery) - task id: {task.id}']
            return {'ok': True, 'message': 'Scheduled sequential training for BOTH processes (FCC -> CCR)'}
        except Exception:
            # fallback: run sequentially in background thread
            background_tasks.add_task(_run_training_sequence, request.n_trials, request.use_optuna)
            return {'ok': True, 'message': 'Scheduled training for BOTH processes (sequential, fallback)'}

    background_tasks.add_task(_run_training, process, request.n_trials, request.use_optuna)
    return {'ok': True, 'message': f'Started training for {process}'}


class CancelRequest(BaseModel):
    process: str


@app.post('/api/train/cancel')
def cancel_training(req: Optional[CancelRequest] = Body(None), process: Optional[str] = None):
    # Accept either a JSON body with {process: 'FCC'} or a query parameter `?process=FCC`
    if req is not None and getattr(req, 'process', None):
        process = req.process
    if not process:
        raise HTTPException(status_code=400, detail='Missing process parameter (body or query param)')
    process = process.upper()
    if process == 'BOTH':
        responses = []
        # If a single 'BOTH' job exists with a celery task id, revoke that first
        if 'BOTH' in TRAIN_JOBS and TRAIN_JOBS['BOTH'].get('celery_task_id'):
            task_id = TRAIN_JOBS['BOTH'].get('celery_task_id')
            try:
                from celery.result import AsyncResult
                res = AsyncResult(task_id)
                res.revoke(terminate=True)
            except Exception:
                pass
            TRAIN_JOBS['BOTH']['running'] = False
            append_log('BOTH', 'Training canceled (celery sequence)')
            publish_update('BOTH', {'event': 'canceled', 'message': f'Canceled celery task {task_id}'})
            responses.append({'ok': True, 'process': 'BOTH', 'message': f'Cancelled sequence task {task_id}'})

        for p in ['FCC', 'CCR']:
            if p not in TRAIN_JOBS:
                responses.append({'ok': False, 'process': p, 'error': 'No such training'})
                continue
            job = TRAIN_JOBS[p]
            task_id = job.get('celery_task_id')
            if not task_id:
                TRAIN_JOBS[p]['running'] = False
                append_log(p, 'Training canceled (no celery task)')
                publish_update(p, {'event': 'canceled', 'message': 'Local training canceled'})
                responses.append({'ok': True, 'process': p, 'message': 'Canceled local training'})
                continue
            try:
                from celery.result import AsyncResult
                res = AsyncResult(task_id)
                res.revoke(terminate=True)
            except Exception:
                pass
            TRAIN_JOBS[p]['running'] = False
            append_log(p, 'Training canceled (celery)')
            publish_update(p, {'event': 'canceled', 'message': f'Canceled celery task {task_id}'})
            responses.append({'ok': True, 'process': p, 'message': f'Cancelled training for {p}'})
        return {'ok': True, 'responses': responses}
    if process not in TRAIN_JOBS:
        raise HTTPException(status_code=404, detail='No such training in progress')
    job = TRAIN_JOBS[process]
    task_id = job.get('celery_task_id')
    if not task_id:
        # If no celery id, we may have a local job; set running false
        TRAIN_JOBS[process]['running'] = False
        append_log(process, 'Training canceled (no celery task)')
        publish_update(process, {'event': 'canceled', 'message': 'Local training canceled'})
        return {'ok': True, 'message': 'Canceled local training'}
    # Revoke Celery task
    try:
        from celery import Celery
        from celery.result import AsyncResult
        res = AsyncResult(task_id)
        res.revoke(terminate=True)
    except Exception:
        pass
    TRAIN_JOBS[process]['running'] = False
    append_log(process, 'Training canceled (celery)')
    publish_update(process, {'event': 'canceled', 'message': f'Canceled celery task {task_id}'})
    return {'ok': True, 'message': f'Cancelled training for {process}'}


@app.get('/api/train/status', response_model=TrainStatus)
def train_status(process: str):
    process = process.upper()
    # First, check in-memory job summary
    job = TRAIN_JOBS.get(process)
    # If we have a live in-memory job running, prefer it (avoid DB regressions)
    if job and job.get('running'):
        return TrainStatus(process=process, running=job.get('running', False), progress=job.get('progress', 0.0), last_update=str(job.get('last_update', time.time())), metrics=job.get('metrics'))
    # Otherwise fall back to DB progress if available
    try:
        db = SessionLocal()
        tr = db.query(TrainingRun).filter(TrainingRun.process == process).order_by(TrainingRun.start_ts.desc()).first()
        if tr:
            progress = float(tr.progress or 0.0)
            running = tr.status in ('running', 'queued')
            metrics = json.loads(tr.metrics) if tr.metrics else None
            return TrainStatus(process=process, running=running, progress=progress, last_update=str(tr.end_ts or tr.start_ts), metrics=metrics)
    except Exception:
        pass
    # No in-memory job, no DB run; return not running
    return TrainStatus(process=process, running=False, progress=0.0, last_update=str(time.time()), metrics=None)


@app.get('/api/models')
def api_list_models():
    return {'models': list_models()}


@app.delete('/api/models')
def api_delete_model(path: str = None, process: str = None, delete_all: bool = False):
    """Delete a specific model (by absolute path), all models for a process, or all models."""
    if delete_all:
        deleted = delete_all_models()
        return {'ok': True, 'deleted': deleted}
    if path:
        ok = delete_model_by_path(path)
        return {'ok': ok, 'deleted': 1 if ok else 0}
    if process:
        d = delete_models_for_process(process)
        return {'ok': True, 'deleted': d}
    raise HTTPException(status_code=400, detail='Specify path, process, or delete_all')


@app.get('/api/logs')
def get_logs(process: str):
    process = process.upper()
    if process not in TRAIN_LOGS:
        return {'logs': []}
    return {'logs': TRAIN_LOGS[process]}


@app.websocket('/ws/training/{process}')
async def websocket_training(ws: WebSocket, process: str):
    await ws.accept()
    # Use redis asyncio client to subscribe to updates
    try:
        import redis.asyncio as redis_async
        redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
        client = redis_async.from_url(redis_url)
        ch_name = f"training:{process.upper()}"
        pubsub = client.pubsub()
        await pubsub.subscribe(ch_name)
        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg.get('data'):
                    data = msg['data']
                    if isinstance(data, bytes):
                        data = data.decode('utf8')
                    # Try to parse JSON and update in-memory TRAIN_JOBS progress if present
                    try:
                        j = json.loads(data)
                        p = j.get('progress')
                        if p is not None:
                            # ensure TRAIN_JOBS entry exists
                            if process.upper() not in TRAIN_JOBS:
                                TRAIN_JOBS[process.upper()] = {'running': True, 'progress': float(p), 'start_ts': time.time(), 'metrics': {}}
                            else:
                                TRAIN_JOBS[process.upper()]['progress'] = float(p)
                                TRAIN_JOBS[process.upper()]['last_update'] = time.time()
                        # handle events like completed/error/canceled and log events
                        evt = j.get('event')
                        if evt in ('completed', 'error', 'canceled'):
                            if process.upper() not in TRAIN_JOBS:
                                TRAIN_JOBS[process.upper()] = {'running': False, 'progress': 1.0 if evt=='completed' else 0.0, 'start_ts': time.time(), 'metrics': {}}
                            else:
                                TRAIN_JOBS[process.upper()]['running'] = False
                                TRAIN_JOBS[process.upper()]['progress'] = 1.0 if evt=='completed' else 0.0
                                TRAIN_JOBS[process.upper()]['last_update'] = time.time()
                            # set metrics if present
                            if j.get('metrics'):
                                TRAIN_JOBS[process.upper()]['metrics'] = j.get('metrics')
                        # handle 'log' event published by worker log handler
                        if j.get('event') == 'log' or j.get('event') == 'log_message':
                            # append raw worker log to TRAIN_LOGS
                            append_log(process.upper(), f"[{j.get('level', '')}] {j.get('message')}")
                    except Exception:
                        # not JSON; ignore
                        pass
                    await ws.send_text(data)
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(ch_name)
    except WebSocketDisconnect:
        return


@app.get('/api/config')
def api_config():
    # Return minimal config summary
    try:
        import config.model_config as mc
        info = {
            'n_trials': mc.RECOMMENDED_N_TRIALS,
            'cv_folds': mc.RECOMMENDED_CV_FOLDS,
            'num_physical_cores': mc.NUM_PHYSICAL_CORES,
            'gpu_available': mc.GPU_AVAILABLE,
        }
        return info
    except Exception as e:
        return {'error': str(e)}


from pydantic import BaseModel
import pandas as pd

@app.post('/api/predict')
def predict(req: PredictRequest):
    import logging
    logging.getLogger(__name__).info('Predict request: %s', req.dict())
    process = req.process.upper()
    if process not in ['FCC', 'CCR']:
        raise HTTPException(status_code=400, detail='Invalid process - choose FCC or CCR')

    # Attempt to load checkpoint with feature_cols
    ckpt = load_checkpoint(process, 'results')
    feature_cols = None
    imputation_values = None
    if ckpt is not None:
        feature_cols = ckpt.get('feature_cols')
        imputation_values = ckpt.get('imputation_values')

    # Load models
    # The model manager expects saved models in /models
    res = {'predictions': []}
    # Build dataframe from records
    df = pd.DataFrame(req.records)
    # Align features
    try:
        from .data_processing import create_basic_physics_features
        df = create_basic_physics_features(df)
    except Exception:
        pass

    if feature_cols is not None:
        df = add_feature_defaults(df, feature_cols)
        df = df[feature_cols]

    # Load models
    from .model_manager import load_model
    pci_models = {}
    h2_models = {}
    for m in ['xgboost', 'lightgbm', 'catboost', 'ridge']:
        m_obj = load_model(process, 'PCI', m)
        if m_obj is not None:
            pci_models[m] = m_obj
        m_obj = load_model(process, 'H2', m)
        if m_obj is not None:
            h2_models[m] = m_obj

    if not pci_models and not h2_models:
        raise HTTPException(status_code=400, detail='No models available for this process - train first')

    models = {
        'pci': {'models': pci_models},
        'h2': {'models': h2_models},
        'feature_cols': feature_cols or df.columns.tolist(),
        'imputation_values': imputation_values or {}
    }
    preds = predict_with_ensemble(models, df.copy())

    for idx, row in preds.iterrows():
        res['predictions'].append({'PCI': float(row['PCI']), 'H2': float(row['H2'])})

    return res
