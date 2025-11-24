from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health():
    r = client.get('/api/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'ok'


def test_models_list():
    r = client.get('/api/models')
    assert r.status_code == 200
    assert 'models' in r.json()


def test_config():
    r = client.get('/api/config')
    assert r.status_code == 200
    data = r.json()
    assert 'n_trials' in data


def test_train_both():
    r = client.post('/api/train', json={'process': 'BOTH', 'n_trials': 2, 'use_optuna': False})
    assert r.status_code == 200
    body = r.json()
    assert body['ok'] is True


def test_cancel_training():
    # Simulate a scheduled celery job
    from src.app import TRAIN_JOBS
    TRAIN_JOBS['FCC'] = {'running': True, 'celery_task_id': 'fake-task-1'}
    r = client.post('/api/train/cancel', json={'process': 'FCC'})
    assert r.status_code == 200
    assert r.json()['ok'] is True
    # reset
    TRAIN_JOBS.pop('FCC', None)
