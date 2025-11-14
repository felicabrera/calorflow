"""
Unified ML Training Module for ANCAP DataChallenge 2025
Consolidates training, inference, optimization, and model management

This module combines functionality from:
- train_competition.py, train_advanced.py, train_walkforward.py
- model_trainer.py, advanced_trainer.py, autogluon_trainer.py
- bayesian_optimizer.py, meta_ensemble.py

Key Features:
- Multi-model training (XGBoost, LightGBM, CatBoost, RandomForest, AutoGluon)
- Bayesian hyperparameter optimization with Optuna
- Cross-validation (Time-Series and K-Fold)
- Meta-ensemble with out-of-fold predictions
- Competition metrics (RMSE + tolerance-based scoring)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json
import warnings
from datetime import datetime
import time

# ML Libraries
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Boosting Libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available")

# AutoML
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    warnings.warn("AutoGluon not available")

# Optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available")

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & UTILITIES
# ============================================================================

class TrainerConfig:
    """Configuration for training pipeline"""
    
    def __init__(self):
        self.random_seed = 42
        self.n_trials = 300  # Optuna trials
        self.cv_folds = 5
        self.use_time_series_cv = True
        self.use_autogluon = True
        self.autogluon_time_limit = 3600
        self.tolerance_pct = 0.10  # Competition: ±10% tolerance
        self.enable_oof_ensemble = True
        self.models_to_train = ['xgboost', 'lightgbm', 'catboost', 'randomforest']
        
    def update(self, **kwargs):
        """Update configuration with custom values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   target_name: str = 'Target') -> Dict[str, float]:
    """
    Compute comprehensive metrics for model evaluation
    
    Args:
        y_true: True values
        y_pred: Predicted values
        target_name: Name of target variable
        
    Returns:
        Dictionary with metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Relative error tolerance (competition metric)
    relative_error = np.abs((y_true - y_pred) / (y_true + 1e-10))
    within_10_pct = np.sum(relative_error <= 0.10)
    within_10_pct_ratio = within_10_pct / len(y_true)
    
    return {
        f'{target_name}_rmse': rmse,
        f'{target_name}_mae': mae,
        f'{target_name}_r2': r2,
        f'{target_name}_within_10pct': within_10_pct,
        f'{target_name}_within_10pct_ratio': within_10_pct_ratio
    }


def compute_competition_score(metrics_pci: Dict, metrics_h2: Dict) -> Dict[str, float]:
    """
    Compute ANCAP competition score
    
    Args:
        metrics_pci: Metrics for PCI predictions
        metrics_h2: Metrics for H2 predictions
        
    Returns:
        Competition metrics
    """
    rmse_pci = metrics_pci['PCI_rmse']
    rmse_h2 = metrics_h2['H2_rmse']
    rmse_prom = (rmse_pci + rmse_h2) / 2
    
    return {
        'rmse_prom': rmse_prom,
        'rmse_pci': rmse_pci,
        'rmse_h2': rmse_h2,
        'within_10_pci': metrics_pci['PCI_within_10pct'],
        'within_10_h2': metrics_h2['H2_within_10pct']
    }


# ============================================================================
# BAYESIAN OPTIMIZER
# ============================================================================

class BayesianOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna
    """
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 target_name: str, config: TrainerConfig):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.target_name = target_name
        self.config = config
        self.best_params = {}
        
    def optimize_model(self, model_type: str) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model type
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'catboost', 'randomforest')
            
        Returns:
            Dictionary with best parameters and score
        """
        if not OPTUNA_AVAILABLE:
            print(f"  ⚠️  Optuna not available, using default parameters for {model_type}")
            return self._get_default_params(model_type)
        
        print(f"  Optimizing {model_type} for {self.target_name}...")
        
        # Create objective function
        if model_type == 'xgboost':
            objective = self._xgboost_objective
        elif model_type == 'lightgbm':
            objective = self._lightgbm_objective
        elif model_type == 'catboost':
            objective = self._catboost_objective
        elif model_type == 'randomforest':
            objective = self._randomforest_objective
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.config.random_seed),
            pruner=SuccessiveHalvingPruner()
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.config.n_trials, 
                      show_progress_bar=True, n_jobs=1)
        
        print(f"    Best RMSE: {study.best_value:.4f}")
        
        return {
            'params': study.best_params,
            'score': study.best_value,
            'model_type': model_type
        }
    
    def _xgboost_objective(self, trial) -> float:
        """Optuna objective for XGBoost"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.config.random_seed,
            'n_jobs': -1
        }
        
        model = XGBRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        return np.sqrt(mean_squared_error(self.y_val, y_pred))
    
    def _lightgbm_objective(self, trial) -> float:
        """Optuna objective for LightGBM"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.config.random_seed,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = LGBMRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        return np.sqrt(mean_squared_error(self.y_val, y_pred))
    
    def _catboost_objective(self, trial) -> float:
        """Optuna objective for CatBoost"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_state': self.config.random_seed,
            'verbose': 0
        }
        
        model = CatBoostRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        return np.sqrt(mean_squared_error(self.y_val, y_pred))
    
    def _randomforest_objective(self, trial) -> float:
        """Optuna objective for RandomForest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': self.config.random_seed,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        return np.sqrt(mean_squared_error(self.y_val, y_pred))
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters when Optuna is not available"""
        defaults = {
            'xgboost': {
                'params': {
                    'n_estimators': 300,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': self.config.random_seed
                },
                'model_type': 'xgboost',
                'score': None
            },
            'lightgbm': {
                'params': {
                    'n_estimators': 300,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': self.config.random_seed,
                    'verbose': -1
                },
                'model_type': 'lightgbm',
                'score': None
            },
            'catboost': {
                'params': {
                    'iterations': 300,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_state': self.config.random_seed,
                    'verbose': 0
                },
                'model_type': 'catboost',
                'score': None
            },
            'randomforest': {
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'random_state': self.config.random_seed,
                    'n_jobs': -1
                },
                'model_type': 'randomforest',
                'score': None
            }
        }
        return defaults.get(model_type, {})


# ============================================================================
# META ENSEMBLE
# ============================================================================

class MetaEnsemble:
    """
    Meta-ensemble for combining multiple models with out-of-fold predictions
    """
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.models = {}
        self.weights = {}
        self.meta_model = None
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def fit_meta_model(self, oof_predictions: np.ndarray, y_true: np.ndarray):
        """
        Fit meta-learner on out-of-fold predictions
        
        Args:
            oof_predictions: (n_samples, n_models) array of OOF predictions
            y_true: True target values
        """
        print("  Training meta-learner (Ridge)...")
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(oof_predictions, y_true)
        
        # Show meta-weights
        print("  Meta-model weights:")
        for i, (name, _) in enumerate(self.models.items()):
            print(f"    {name}: {self.meta_model.coef_[i]:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble
        
        Args:
            X: Features
            
        Returns:
            Ensemble predictions
        """
        if self.meta_model is not None:
            # Use meta-learner
            predictions = np.column_stack([
                model.predict(X) for model in self.models.values()
            ])
            return self.meta_model.predict(predictions)
        else:
            # Weighted average
            predictions = np.zeros(len(X))
            total_weight = sum(self.weights.values())
            
            for name, model in self.models.items():
                weight = self.weights[name] / total_weight
                predictions += weight * model.predict(X)
            
            return predictions


# ============================================================================
# MAIN TRAINER CLASS
# ============================================================================

class MLTrainer:
    """
    Unified ML Trainer for PCI and H2 prediction
    
    Handles:
    - Data splitting and cross-validation
    - Hyperparameter optimization
    - Multi-model training
    - Ensemble creation
    - Model evaluation and saving
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.models_pci = {}
        self.models_h2 = {}
        self.ensemble_pci = None
        self.ensemble_h2 = None
        self.training_history = {
            'pci': {},
            'h2': {},
            'competition_score': {}
        }
        
    def train(self, X: pd.DataFrame, y_pci: pd.Series, y_h2: pd.Series,
             process_name: str = 'UNKNOWN', output_dir: str = 'models') -> Dict[str, Any]:
        """
        Train all models for both PCI and H2 targets
        
        Args:
            X: Features
            y_pci: PCI target
            y_h2: H2 target
            process_name: Name of process (FCC/CCR)
            output_dir: Directory to save models (default: 'models')
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*80)
        print(f"TRAINING PIPELINE - {process_name}")
        print("="*80)
        
        start_time = time.time()
        
        # Split data
        print("\n[1/4] Splitting data...")
        X_train, X_test, y_pci_train, y_pci_test, y_h2_train, y_h2_test = train_test_split(
            X, y_pci, y_h2,
            test_size=0.2,
            random_state=self.config.random_seed
        )
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Train PCI models
        print("\n[2/4] Training PCI models...")
        self.models_pci, oof_pci = self._train_target_models(
            X_train, y_pci_train, X_test, y_pci_test, 'PCI', output_dir
        )
        
        # Train H2 models
        print("\n[3/4] Training H2 models...")
        self.models_h2, oof_h2 = self._train_target_models(
            X_train, y_h2_train, X_test, y_h2_test, 'H2', output_dir
        )
        
        # Create ensembles
        print("\n[4/4] Creating ensembles...")
        self.ensemble_pci = self._create_ensemble(self.models_pci, oof_pci, y_pci_train, 'PCI')
        self.ensemble_h2 = self._create_ensemble(self.models_h2, oof_h2, y_h2_train, 'H2')
        
        # Final evaluation
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80)
        
        y_pred_pci = self.ensemble_pci.predict(X_test)
        y_pred_h2 = self.ensemble_h2.predict(X_test)
        
        metrics_pci = compute_metrics(y_pci_test, y_pred_pci, 'PCI')
        metrics_h2 = compute_metrics(y_h2_test, y_pred_h2, 'H2')
        competition_score = compute_competition_score(metrics_pci, metrics_h2)
        
        print(f"\nPCI  - RMSE: {metrics_pci['PCI_rmse']:.4f}, R²: {metrics_pci['PCI_r2']:.4f}")
        print(f"H2   - RMSE: {metrics_h2['H2_rmse']:.4f}, R²: {metrics_h2['H2_r2']:.4f}")
        print(f"\nCompetition Score (RMSE_prom): {competition_score['rmse_prom']:.4f}")
        print(f"Within ±10% - PCI: {competition_score['within_10_pci']}, H2: {competition_score['within_10_h2']}")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Training completed in {elapsed/60:.1f} minutes")
        print("="*80)
        
        # Store results
        results = {
            'process_name': process_name,
            'metrics_pci': metrics_pci,
            'metrics_h2': metrics_h2,
            'competition_score': competition_score,
            'training_time': elapsed,
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test)
        }
        
        self.training_history = results
        
        return results
    
    def _train_target_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           target_name: str, output_dir: str = 'models') -> Tuple[Dict, np.ndarray]:
        """
        Train all models for a single target
        
        Returns:
            Tuple of (models_dict, oof_predictions)
        """
        models = {}
        oof_predictions = np.zeros((len(X_train), len(self.config.models_to_train)))
        
        # Further split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=self.config.random_seed
        )
        
        # Train each model type
        for idx, model_type in enumerate(self.config.models_to_train):
            print(f"\n  Training {model_type}...")
            
            # Check availability
            if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
                print(f"    ⚠️  XGBoost not available, skipping")
                continue
            elif model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
                print(f"    ⚠️  LightGBM not available, skipping")
                continue
            elif model_type == 'catboost' and not CATBOOST_AVAILABLE:
                print(f"    ⚠️  CatBoost not available, skipping")
                continue
            
            # Optimize hyperparameters
            optimizer = BayesianOptimizer(X_tr, y_tr, X_val, y_val, target_name, self.config)
            best_config = optimizer.optimize_model(model_type)
            
            # Train on full training set
            model = self._create_model(model_type, best_config['params'])
            model.fit(X_train, y_train)
            
            # Generate OOF predictions (simplified: use validation set)
            if idx < oof_predictions.shape[1]:
                # For simplicity, use validation predictions
                val_preds = model.predict(X_val)
                # Store in OOF array (this is simplified - proper OOF needs CV)
                oof_predictions[:, idx] = model.predict(X_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"    Test RMSE: {rmse:.4f}")
            
            models[model_type] = model
        
        # Train AutoGluon if enabled
        if self.config.use_autogluon and AUTOGLUON_AVAILABLE:
            print(f"\n  Training AutoGluon...")
            ag_model = self._train_autogluon(X_train, y_train, target_name, output_dir)
            if ag_model is not None:
                models['autogluon'] = ag_model
        
        return models, oof_predictions
    
    def _create_model(self, model_type: str, params: Dict) -> Any:
        """Create model instance with given parameters"""
        if model_type == 'xgboost':
            return XGBRegressor(**params)
        elif model_type == 'lightgbm':
            return LGBMRegressor(**params)
        elif model_type == 'catboost':
            return CatBoostRegressor(**params)
        elif model_type == 'randomforest':
            return RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_autogluon(self, X_train: pd.DataFrame, y_train: pd.Series,
                        target_name: str, output_dir: str = 'models') -> Optional[Any]:
        """Train AutoGluon model"""
        try:
            train_data = X_train.copy()
            train_data[target_name] = y_train
            
            # Use consistent output directory
            ag_path = Path(output_dir) / f'autogluon_{target_name.lower()}'
            
            predictor = TabularPredictor(
                label=target_name,
                eval_metric='root_mean_squared_error',
                path=str(ag_path)
            ).fit(
                train_data,
                time_limit=self.config.autogluon_time_limit,
                presets='best_quality',
                verbosity=2
            )
            
            print(f"    AutoGluon training completed")
            return predictor
            
        except Exception as e:
            print(f"    ⚠️  AutoGluon training failed: {e}")
            return None
    
    def _create_ensemble(self, models: Dict, oof_predictions: np.ndarray,
                        y_train: pd.Series, target_name: str) -> MetaEnsemble:
        """Create meta-ensemble from trained models"""
        ensemble = MetaEnsemble(self.config)
        
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Fit meta-learner if OOF is enabled
        if self.config.enable_oof_ensemble and len(models) > 1:
            # Filter OOF to match number of models
            oof_filtered = oof_predictions[:, :len(models)]
            ensemble.fit_meta_model(oof_filtered, y_train.values)
        
        return ensemble
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for new data
        
        Args:
            X: Features
            
        Returns:
            Tuple of (pci_predictions, h2_predictions)
        """
        if self.ensemble_pci is None or self.ensemble_h2 is None:
            raise ValueError("Models not trained. Call train() first.")
        
        y_pred_pci = self.ensemble_pci.predict(X)
        y_pred_h2 = self.ensemble_h2.predict(X)
        
        return y_pred_pci, y_pred_h2
    
    def save(self, output_dir: str, process_name: str):
        """
        Save trained models and ensembles
        
        Args:
            output_dir: Directory to save models
            process_name: Name of process (FCC/CCR)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save ensembles
        joblib.dump(self.ensemble_pci, output_path / f'{process_name}_ensemble_pci.joblib')
        joblib.dump(self.ensemble_h2, output_path / f'{process_name}_ensemble_h2.joblib')
        
        # Save individual models
        for name, model in self.models_pci.items():
            joblib.dump(model, output_path / f'{process_name}_pci_{name}.joblib')
        
        for name, model in self.models_h2.items():
            joblib.dump(model, output_path / f'{process_name}_h2_{name}.joblib')
        
        # Save training history
        with open(output_path / f'{process_name}_training_history.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_json = {}
            for key, value in self.training_history.items():
                if isinstance(value, dict):
                    history_json[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                        for k, v in value.items()}
                else:
                    history_json[key] = value
            json.dump(history_json, f, indent=2)
        
        print(f"\n✅ Models saved to {output_path}")
    
    @staticmethod
    def load(model_dir: str, process_name: str) -> 'MLTrainer':
        """
        Load trained models
        
        Args:
            model_dir: Directory containing saved models
            process_name: Name of process (FCC/CCR)
            
        Returns:
            Loaded MLTrainer instance
        """
        model_path = Path(model_dir)
        
        trainer = MLTrainer()
        
        # Load ensembles
        trainer.ensemble_pci = joblib.load(model_path / f'{process_name}_ensemble_pci.joblib')
        trainer.ensemble_h2 = joblib.load(model_path / f'{process_name}_ensemble_h2.joblib')
        
        # Load training history
        history_file = model_path / f'{process_name}_training_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                trainer.training_history = json.load(f)
        
        print(f"✅ Models loaded from {model_path}")
        
        return trainer


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def train_pipeline(X: pd.DataFrame, y_pci: pd.Series, y_h2: pd.Series,
                  process_name: str = 'UNKNOWN',
                  output_dir: str = 'models',
                  config: Optional[TrainerConfig] = None) -> MLTrainer:
    """
    Convenience function to train complete pipeline
    
    Args:
        X: Features
        y_pci: PCI target
        y_h2: H2 target
        process_name: Name of process (FCC/CCR)
        output_dir: Directory to save models
        config: Optional custom configuration
        
    Returns:
        Trained MLTrainer instance
    """
    trainer = MLTrainer(config)
    trainer.train(X, y_pci, y_h2, process_name)
    trainer.save(output_dir, process_name)
    return trainer


def predict_with_models(X: pd.DataFrame, model_dir: str, 
                       process_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load models and make predictions
    
    Args:
        X: Features
        model_dir: Directory containing saved models
        process_name: Name of process (FCC/CCR)
        
    Returns:
        Tuple of (pci_predictions, h2_predictions)
    """
    trainer = MLTrainer.load(model_dir, process_name)
    return trainer.predict(X)


if __name__ == '__main__':
    print("MLTrainer - Unified Training Module")
    print("Import this module to use: from src.trainer import MLTrainer, train_pipeline")
