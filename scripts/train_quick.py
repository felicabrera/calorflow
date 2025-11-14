import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainer import train_pipeline, TrainerConfig
from src.data_utils import load_fcc_data, merge_operational_and_gas, preprocess_data
from src.features import create_features, clean_feature_names
import pandas as pd
import numpy as np

print("="*80)
print("QUICK TRAINING - FCC Process")
print("="*80)

print("\n[1/4] Loading FCC data...")
df_operational, df_gas = load_fcc_data('data/FCC - Cracking Catalítico')

print("\n[2/4] Merging and preprocessing...")
df = merge_operational_and_gas(df_operational, df_gas)
df = preprocess_data(df, handle_missing='interpolate', remove_outliers=True)

print("\n[3/4] Creating features...")
df = create_features(df)
df = clean_feature_names(df)

print("\n[4/4] Training models...")
X = df.drop(['PCI', 'H2'], axis=1, errors='ignore')
y_pci = df['PCI']
y_h2 = df['H2']

config = TrainerConfig(
    random_seed=42,
    n_trials=50,
    cv_folds=3,
    use_autogluon=False,
    models_to_train=['xgboost', 'lightgbm']
)

trainer = train_pipeline(
    X=X,
    y_pci=y_pci,
    y_h2=y_h2,
    process_name='FCC',
    output_dir='models',
    config=config
)

print("\n"+"="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Models saved to: models/")
print("\nYou can now start the API and test with Postman:")
print("  python scripts/run_api.py")
print("\nPostman endpoint:")
print("  POST http://localhost:8000/predict")
print("="*80)
