from src import model_manager as mm
from pathlib import Path
import joblib


def test_save_and_list_model(tmp_path):
    # Create a dummy model (Ridge)
    from sklearn.linear_model import Ridge
    m = Ridge()
    p = Path('models')
    # ensure clean models dir for test
    if p.exists():
        for f in p.glob('**/*'):
            try:
                f.unlink()
            except Exception:
                pass

    path = mm.save_model('TESTPROC', 'PCI', 'ridge', m)
    assert path.exists()
    models = mm.list_models()
    assert any('TESTPROC' in m['file'] or 'testproc' in m['path'] for m in models)
