"""
Model manager: save/load models and metadata, checkpoint logic
"""
from pathlib import Path
import joblib
import json
from datetime import datetime

MODELS_DIR = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def model_path(process_name: str, target: str, model_type: str, timestamp: str = None) -> Path:
    ts = timestamp or ''
    name = f"{process_name.lower()}_{target.lower()}_{model_type}_{ts}.joblib" if ts else f"{process_name.lower()}_{target.lower()}_{model_type}.joblib"
    return MODELS_DIR / process_name.lower() / name


def save_model(process_name: str, target: str, model_type: str, model_obj):
    # create subdir
    proc_dir = MODELS_DIR / process_name.lower()
    proc_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    path = model_path(process_name, target, model_type, timestamp)
    joblib.dump(model_obj, path)
    return path


def load_model(process_name: str, target: str, model_type: str):
    # Try exact path (no timestamp)
    path = model_path(process_name, target, model_type)
    if path.exists():
        return joblib.load(path)
    # Otherwise find the latest timestamped file matching pattern
    proc_dir = MODELS_DIR / process_name.lower()
    if not proc_dir.exists():
        return None
    pattern = f"{process_name.lower()}_{target.lower()}_{model_type}_*.joblib"
    candidates = sorted(proc_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return joblib.load(candidates[0])


def list_models():
    results = []
    for p in MODELS_DIR.glob('**/*.joblib'):
        results.append({'file': p.name, 'path': str(p.resolve()), 'modified': datetime.fromtimestamp(p.stat().st_mtime).isoformat()})
    return results


def delete_model_by_path(path_str: str) -> bool:
    p = Path(path_str)
    try:
        if p.exists() and p.suffix == '.joblib':
            p.unlink()
            return True
        return False
    except Exception:
        return False


def delete_models_for_process(process_name: str) -> int:
    proc_dir = MODELS_DIR / process_name.lower()
    deleted = 0
    if proc_dir.exists():
        for p in proc_dir.glob('**/*.joblib'):
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass
    return deleted


def delete_all_models() -> int:
    deleted = 0
    for p in MODELS_DIR.glob('**/*.joblib'):
        try:
            p.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted


def save_checkpoint(process_name, checkpoint_type, data: dict):
    ckpt_file = CHECKPOINT_DIR / f"{process_name.lower()}_{checkpoint_type}.json"
    with open(ckpt_file, 'w', encoding='utf8') as fh:
        json.dump(data, fh, default=str, indent=2)
    return ckpt_file


def load_checkpoint(process_name, checkpoint_type='results'):
    ckpt_file = CHECKPOINT_DIR / f"{process_name.lower()}_{checkpoint_type}.json"
    if not ckpt_file.exists():
        return None
    with open(ckpt_file, 'r', encoding='utf8') as fh:
        return json.load(fh)


def checkpoint_exists(process_name, checkpoint_type='results'):
    ckpt_file = CHECKPOINT_DIR / f"{process_name.lower()}_{checkpoint_type}.json"
    return ckpt_file.exists()
