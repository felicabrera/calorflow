"""
Simplified training functions and API-friendly wrappers.
Keeps core logic: Train PCI and H2 ensembles (xgb, lgb, cat) and return summary metrics.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Optional imports
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    import optuna
except Exception:
    optuna = None

try:
    import mlflow
except Exception:
    mlflow = None

import os

# Configure MLflow to use remote tracking server if provided via env
try:
    if mlflow is not None:
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            # print a known message in logs for easier debugging
            import logging
            logging.getLogger(__name__).info('MLflow tracking URI set to %s', mlflow_tracking_uri)
except Exception:
    pass


def _train_single_model(X, y, model_name: str):
    """Train a single model with simple default params (no Optuna).
    model_name in ['xgboost','lightgbm','catboost','ridge']
    """
    if model_name == 'xgboost' and XGBRegressor is not None:
        model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=1, verbosity=0)
        model.fit(X, y)
        return model
    if model_name == 'lightgbm' and LGBMRegressor is not None:
        model = LGBMRegressor(n_estimators=300, random_state=42, n_jobs=1)
        model.fit(X, y)
        return model
    if model_name == 'catboost' and CatBoostRegressor is not None:
        model = CatBoostRegressor(iterations=200, depth=8, learning_rate=0.05, verbose=False, random_state=42)
        model.fit(X, y)
        return model
    # Fallback: Ridge
    model = Ridge(alpha=1.0, random_state=42, max_iter=10000)
    model.fit(X, y)
    return model


def train_model_for_target(X_train: pd.DataFrame, y_train: pd.Series, target_name: str, n_trials: int = 20, use_optuna: bool = False, progress_callback=None) -> Dict[str, Any]:
    """Train ensemble of models for a single target (PCI or H2).
    For the API, we keep a small default n_trials and a fallback to basic training.
    """
    # Split small validation set
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    models = {}
    metrics = {}

    # If optuna requested, perform a small tune for XGB only (if available)
    if use_optuna and optuna is not None and X_train.shape[0] > 50:
        def objective(trial):
            n_est = trial.suggest_int('n_estimators', 100, 500)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            lr = trial.suggest_loguniform('learning_rate', 0.01, 0.2)
            from sklearn.model_selection import train_test_split
            Xtr, Xv, ytr, yv = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            if XGBRegressor is not None:
                m = XGBRegressor(n_estimators=n_est, max_depth=max_depth, learning_rate=lr, random_state=42, n_jobs=1, verbosity=0)
                m.fit(Xtr, ytr)
                pred = m.predict(Xv)
                return float(np.sqrt(mean_squared_error(yv, pred)))
            else:
                return float(np.inf)

        def _optuna_cb(study, trial):
            # publish trial progress
            try:
                trial_idx = trial.number + 1
                total = min(n_trials, 20)
                if progress_callback:
                    progress = 0.2 * (trial_idx / float(total))  # optuna phase weight 20%
                    progress_callback({'event': 'optuna_trial', 'trial': trial_idx, 'value': trial.value, 'progress': progress})
            except Exception:
                pass

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=min(n_trials, 20), callbacks=[_optuna_cb])
        best = study.best_params
        # Use best params for XGBoost training later
        xgb_best_params = best
    models_total = 3
    for idx, model_name in enumerate(['xgboost', 'lightgbm', 'catboost']):
        try:
            m = _train_single_model(X_tr, y_tr, model_name)
            models[model_name] = m
        except Exception as e:
            # Fallback to ridge
            models[model_name] = _train_single_model(X_tr, y_tr, 'ridge')
        # publish training progress for each model trained
        try:
            if progress_callback:
                # stage: training models phase accounts for remaining 80% (0.2 -> 1.0)
                stage_progress = 0.2 + 0.8 * ((idx + 1) / float(models_total))
                progress_callback({'event': 'model_trained', 'model': model_name, 'progress': stage_progress})
        except Exception:
            pass

    # Ensemble prediction
    preds = []
    for m in models.values():
        preds.append(m.predict(X_val))
    # Average
    ensemble_pred = np.mean(preds, axis=0)

    val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    ss_tot = sum((y_val - y_val.mean())**2)
    if ss_tot < 1e-9:
        val_r2 = 0.0
    else:
        val_r2 = 1.0 - sum((y_val - ensemble_pred)**2) / ss_tot
    pct_within_10 = (np.abs((y_val - ensemble_pred) / (y_val + 1e-6)) <= 0.10).mean() * 100.0

    metrics = {
        'val_rmse': float(val_rmse),
        'val_r2': float(val_r2),
        'pct_within_10': float(pct_within_10)
    }

    # Log to mlflow if available
    try:
        if mlflow is not None:
            mlflow.set_experiment('Calorflow')
            with mlflow.start_run(run_name=f"{target_name}"):
                # log metrics
                mlflow.log_metric('val_rmse', metrics['val_rmse'])
                mlflow.log_metric('val_r2', metrics['val_r2'])
                mlflow.log_metric('pct_within_10', metrics['pct_within_10'])
                # log basic model params
                for i, (mname, m) in enumerate(models.items()):
                    mlflow.log_param(f"model_{i}", mname)
    except Exception:
        pass

    return {
        'models': models,
        'metrics': metrics
    }


def train_process_models(train_df: pd.DataFrame, process_name: str, n_trials: int = 20, use_optuna: bool = False, progress_callback=None):
    """Train PCI and H2 ensembles using training dataframe.
    Returns dictionary with model objects, metrics and feature lists.
    """
    # If train_df is None, try to load from processed data folder
    if train_df is None:
        import os
        from pathlib import Path
        folder = Path('.') / 'data' / 'processed'
        fname = folder / f"{process_name.lower()}_train.csv"
        if fname.exists():
            train_df = pd.read_csv(fname)
        else:
            raise ValueError(f"Processed train file not found: {fname}")

    # determine features
    exclude = ['sampled_date', 'PCI', 'H2', 'sample_weight', 'has_actual_measurement']
    feature_cols = [c for c in train_df.columns if c not in exclude]

    X = train_df[feature_cols].copy()
    y_pci = train_df['PCI'].copy()
    y_h2 = train_df['H2'].copy()

    # Calculate imputation values (medians)
    imputation_values = {}
    for c in X.columns:
        if X[c].isnull().all():
            imputation_values[c] = 0.0
            X[c] = 0.0
        else:
            median_val = float(X[c].median())
            imputation_values[c] = median_val
            X[c] = X[c].fillna(median_val)

    # Create basic physics features (we import function locally to avoid cycles)
    try:
        from .data_processing import create_basic_physics_features
        X = create_basic_physics_features(X)
    except Exception:
        pass

    # Train PCI
    if progress_callback:
        try:
            progress_callback({'event': 'stage', 'stage': 'pci_start', 'progress': 0.05})
        except Exception:
            pass
    pci_res = train_model_for_target(X, y_pci, f"{process_name}_PCI", n_trials=n_trials, use_optuna=use_optuna, progress_callback=progress_callback)
    # Train H2
    if progress_callback:
        try:
            progress_callback({'event': 'stage', 'stage': 'h2_start', 'progress': 0.5})
        except Exception:
            pass
    h2_res = train_model_for_target(X, y_h2, f"{process_name}_H2", n_trials=n_trials, use_optuna=use_optuna, progress_callback=progress_callback)

    if progress_callback:
        try:
            progress_callback({'event': 'completed', 'progress': 1.0, 'metrics': {'pci': pci_res['metrics'], 'h2': h2_res['metrics']}})
        except Exception:
            pass

    return {
        'process_name': process_name,
        'feature_cols': X.columns.tolist(),
        'imputation_values': imputation_values,
        'pci': pci_res,
        'h2': h2_res
    }


# Prediction wrapper

def predict_with_ensemble(models_dict: Dict[str, Any], df: pd.DataFrame):
    feature_cols = models_dict.get('feature_cols') or df.columns.tolist()
    # Basic alignment
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].copy()

    # Impute using saved values if available, else fallback to 0 (safe default for API)
    imputation_values = models_dict.get('imputation_values', {})
    
    for c in X.columns:
        if c in imputation_values:
            X[c] = X[c].fillna(imputation_values[c])
        else:
            # Fallback for unknown columns or if no imputation values provided
            X[c] = X[c].fillna(0)

    pci_models = models_dict['pci']['models']
    h2_models = models_dict['h2']['models']

    # Get preds
    preds_pci = np.mean([m.predict(X) for m in pci_models.values()], axis=0)
    preds_h2 = np.mean([m.predict(X) for m in h2_models.values()], axis=0)

    out = df[['sampled_date']].copy() if 'sampled_date' in df.columns else pd.DataFrame(index=df.index)
    out['PCI'] = preds_pci
    out['H2'] = preds_h2
    return out
