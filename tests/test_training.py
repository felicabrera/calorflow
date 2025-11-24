import pandas as pd
import numpy as np
from src.training import train_process_models


def test_train_process_models_minimal():
    # Small synthetic dataset to exercise pipeline
    n = 100
    df = pd.DataFrame({
        'sampled_date': pd.date_range('2024-01-01', periods=n, freq='H'),
        'sensor_flow_mean': np.random.rand(n) * 100,
        'sensor_temp_mean': np.random.rand(n) * 200,
        'PCI': 8000 + np.random.randn(n) * 50,
        'H2': 1.5 + np.random.randn(n) * 0.1,
    })

    res = train_process_models(df, 'FCC')
    assert 'pci' in res and 'h2' in res
    assert 'metrics' in res['pci'] and 'metrics' in res['h2']
