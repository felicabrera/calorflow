"""
Configuration file for the training pipeline.
Centralized configuration for hyperparameter optimization and model training
"""

import os
import multiprocessing
import warnings

# ============================================================================
# GENERAL SETTINGS
# ============================================================================
RANDOM_SEED = 42
N_ESTIMATORS = 100

# ============================================================================
# HARDWARE AUTO-DETECTION & OPTIMIZATION
# ============================================================================
# Import hardware optimizer
try:
    from src.hardware_optimizer import get_hardware_optimizer
    _HARDWARE_OPTIMIZER = get_hardware_optimizer()
    _HARDWARE_CONFIG = _HARDWARE_OPTIMIZER.get_config_updates()
    HARDWARE_AUTO_DETECTED = True
    
    # Extract hardware specs
    NUM_PHYSICAL_CORES = _HARDWARE_OPTIMIZER.cpu_info['physical_cores']
    GPU_AVAILABLE = _HARDWARE_OPTIMIZER.gpu_info['available']
    GPU_VRAM_GB = _HARDWARE_OPTIMIZER.gpu_info['total_vram_gb']
    
    # Use recommended values
    RECOMMENDED_N_TRIALS = _HARDWARE_CONFIG['OPTUNA_CONFIG']['n_trials']
    RECOMMENDED_TIMEOUT = _HARDWARE_CONFIG['OPTUNA_CONFIG']['timeout']
    RECOMMENDED_CV_FOLDS = _HARDWARE_CONFIG['OPTUNA_CONFIG']['cv_folds']
    RECOMMENDED_N_JOBS = _HARDWARE_CONFIG['OPTUNA_CONFIG']['n_jobs']
    
except Exception as e:
    warnings.warn(f"Hardware auto-detection failed: {e}. Using default values.")
    HARDWARE_AUTO_DETECTED = False
    
    # Fallback to manual detection
    try:
        import psutil
        NUM_PHYSICAL_CORES = psutil.cpu_count(logical=False)
        if NUM_PHYSICAL_CORES is None:
            NUM_PHYSICAL_CORES = psutil.cpu_count()
    except ImportError:
        try:
            NUM_PHYSICAL_CORES = len(os.sched_getaffinity(0)) // 2
        except AttributeError:
            NUM_PHYSICAL_CORES = multiprocessing.cpu_count() // 2
    NUM_PHYSICAL_CORES = max(1, NUM_PHYSICAL_CORES)
    
    # Default values
    GPU_AVAILABLE = False
    GPU_VRAM_GB = 0
    RECOMMENDED_N_TRIALS = 300
    RECOMMENDED_TIMEOUT = 3600
    RECOMMENDED_CV_FOLDS = 3
    RECOMMENDED_N_JOBS = 1

# GPU Configuration
GPU_RECOMMENDED = GPU_AVAILABLE

# ============================================================================
# BAYESIAN OPTIMIZATION SETTINGS (AUTO-OPTIMIZED)
# ============================================================================
OPTUNA_CONFIG = {
    'n_trials': RECOMMENDED_N_TRIALS,
    'timeout': RECOMMENDED_TIMEOUT,
    'n_jobs': RECOMMENDED_N_JOBS,
    'show_progress_bar': True,
    
    # Cross-validation settings
    'cv_folds': RECOMMENDED_CV_FOLDS,
    'cv_scoring': 'neg_root_mean_squared_error',
    
    # Pruning (early stopping of unpromising trials)
    'pruner': 'SuccessiveHalvingPruner',
    'pruner_params': {
        'min_resource': 5,
        'reduction_factor': 3,
        'min_early_stopping_rate': 0
    },
    
    # Sampler (search strategy)
    'sampler': 'TPESampler',
    'sampler_params': {
        'n_startup_trials': 10,
        'multivariate': True,
        'seed': RANDOM_SEED,
        # Advanced TPE settings for competition
        'n_ei_candidates': 50,
        'consider_prior': True,
        'prior_weight': 1.0,
        'consider_magic_clip': True,
        'consider_endpoints': True,
    }
}

# ============================================================================
# ADAPTIVE PARAMETER BOUNDS (Scale with dataset size)
# ============================================================================
def get_adaptive_bounds(n_samples, n_features):
    """
    Calculate adaptive hyperparameter bounds based on dataset size
    
    Args:
        n_samples: Number of training samples
        n_features: Number of features
        
    Returns:
        dict: Adaptive bounds for each model type
    """
    
    # Calculate safe minimums (at least 2% of data or 10 samples)
    min_child_safe = max(10, int(n_samples * 0.02))
    min_samples_safe = max(10, int(n_samples * 0.02))
    min_data_safe = max(5, int(n_samples * 0.02))
    
    # Calculate adaptive maximums (scale with dataset size)
    min_child_max = max(min_child_safe + 10, min(300, n_samples // 5))
    min_samples_max = max(min_samples_safe + 10, min(500, n_samples // 3))
    min_data_max = max(min_data_safe + 10, min(300, n_samples // 5))
    
    # Depth limits based on dataset size
    if n_samples < 500:
        max_depth_limit_xgb = 8
        max_depth_limit_lgb = 8
        max_depth_limit_cat = 8
    elif n_samples < 5000:
        max_depth_limit_xgb = 10
        max_depth_limit_lgb = 10
        max_depth_limit_cat = 10
    else:
        max_depth_limit_xgb = 15
        max_depth_limit_lgb = 15
        max_depth_limit_cat = 12
    
    return {
        'xgboost': {
            'min_child_weight_min': min_child_safe,
            'min_child_weight_max': min_child_max,
            'max_depth_max': max_depth_limit_xgb,
        },
        'lightgbm': {
            'min_child_samples_min': min_samples_safe,
            'min_child_samples_max': min_samples_max,
            'max_depth_max': max_depth_limit_lgb,
            # LightGBM specific
            'max_leaves_safe': max(15, min(300, n_samples // (2 * min_samples_safe))),
        },
        'catboost': {
            'min_data_in_leaf_min': min_data_safe,
            'min_data_in_leaf_max': min_data_max,
            'max_depth_max': max_depth_limit_cat,
        }
    }

# ============================================================================
# XGBOOST HYPERPARAMETER RANGES
# ============================================================================
XGBOOST_PARAM_RANGES = {
    'n_estimators': (200, 1000),
    'max_depth': (3, 8),
    'learning_rate': (0.01, 0.3, 'log'),
    'subsample': (0.4, 0.85),
    'colsample_bytree': (0.4, 0.85),
    'colsample_bylevel': (0.4, 0.85),
    'colsample_bynode': (0.4, 0.85),
    'reg_alpha': (0.1, 500.0, 'log'),
    'reg_lambda': (1.0, 1000.0, 'log'),
    'gamma': (0.01, 20.0, 'log'),
    'min_child_weight': (20, 80),
    'max_delta_step': (0, 8),
    'grow_policy': ['depthwise', 'lossguide'],
}

XGBOOST_FIXED_PARAMS = {
    'random_state': RANDOM_SEED,
    'n_jobs': NUM_PHYSICAL_CORES,
    'verbosity': 0,
    'tree_method': 'gpu_hist',
    'device': 'cuda'
}

# ============================================================================
# LIGHTGBM HYPERPARAMETER RANGES
# ============================================================================
LIGHTGBM_PARAM_RANGES = {
    'n_estimators': (200, 1000),
    'max_depth': (3, 15),
    'num_leaves': (15, 400),
    'learning_rate': (0.0005, 0.15, 'log'),
    'feature_fraction': (0.5, 0.95),
    'bagging_fraction': (0.5, 0.95),
    'bagging_freq': (1, 10),
    'reg_alpha': (0.01, 100.0, 'log'),
    'reg_lambda': (0.1, 200.0, 'log'),
    'min_split_gain': (1e-4, 0.5, 'log'),
    'path_smooth': (0.0, 1.0),
}

LIGHTGBM_FIXED_PARAMS = {
    'random_state': RANDOM_SEED,
    'n_jobs': NUM_PHYSICAL_CORES,
    'num_threads': NUM_PHYSICAL_CORES,
    'verbosity': -1,
    'force_row_wise': True,
    'force_col_wise': True,
    'deterministic': True,
    'device': 'gpu'
}

# ============================================================================
# CATBOOST HYPERPARAMETER RANGES
# ============================================================================
CATBOOST_PARAM_RANGES = {
    'iterations': (100, 500),
    'depth': (4, 10),
    'learning_rate': (0.0005, 0.15, 'log'),
    'l2_leaf_reg': (0.5, 50.0, 'log'),
    'border_count': (32, 128),
    'bagging_temperature': (0.0, 8.0),
    'random_strength': (0.5, 8.0),
    'subsample': (0.5, 0.95),
    'colsample_bylevel': (0.5, 0.95),
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
}

CATBOOST_FIXED_PARAMS = {
    'random_state': RANDOM_SEED,
    'thread_count': NUM_PHYSICAL_CORES,
    'task_type': 'GPU',
    'verbose': False,
    'allow_writing_files': True,
    'train_dir': 'cache/catboost_info',
    'early_stopping_rounds': 50,
}

# ============================================================================
# RANDOM FOREST HYPERPARAMETER RANGES
# ============================================================================
RANDOMFOREST_PARAM_RANGES = {
    # MORE TREES: Random Forest benefits from many trees
    'n_estimators': (500, 2000),
    
    # DEPTH: Deeper trees for complex patterns
    'max_depth': (15, 50),
    
    # SPLITTING: More conservative to prevent overfitting
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10),
    
    # FEATURES: More aggressive sampling
    'max_features': ['sqrt', 'log2', 0.5, 0.6, 0.7, 0.8],
    
    # BOOTSTRAP SAMPLING: More aggressive
    'max_samples': (0.6, 0.95),
}

RANDOMFOREST_FIXED_PARAMS = {
    'bootstrap': True,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}

# ============================================================================
# RIDGE REGRESSION HYPERPARAMETER RANGES
# ============================================================================
RIDGE_PARAM_RANGES = {
    'alpha': (1e-3, 10000.0, 'log'),
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
}

RIDGE_FIXED_PARAMS = {
    'random_state': RANDOM_SEED,
    'max_iter': 50000,
    'tol': 1e-6,
}

# ============================================================================
# MODEL SELECTION
# ============================================================================
MODELS_TO_TRAIN = {
    'xgboost': True,
    'lightgbm': True,
    'catboost': True,
    'randomforest': True,
    'ridge': False
}

# ============================================================================
# SAMPLE WEIGHT CONFIGURATION
# ============================================================================
SAMPLE_WEIGHT_CONFIG = {
    'use_sample_weights': True,
    'actual_measurement_weight': 1.0,
    'interpolated_weight': 0.5,
    'use_recency_weights': True,
    'recency_half_life': 50,
}

# ============================================================================
# VALIDATION STRATEGY
# ============================================================================
VALIDATION_CONFIG = {
    'test_size': 0.2,
    'shuffle': True,
    'stratify': None,  # Set to target column for stratification
    'random_state': RANDOM_SEED,
    # Dataset size thresholds
    'min_dataset_size': 50,  # Minimum samples required for training
    'min_dataset_size_for_config_test_size': 100,  # Use fallback test_size if dataset < this
    'fallback_test_size': 0.15,
    # CV strategy thresholds
    'small_dataset_threshold': 10000,
    'min_train_ratio_timeseries': 0.20,
}

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================
OUTPUT_CONFIG = {
    'models_dir': 'models',
    'plots_dir': 'plots',
    'logs_dir': 'logs',
    'verbose': True,
    'save_plots': True,
    'save_models': True,
}

# ============================================================================
# ENSEMBLE CONFIGURATION
# ============================================================================
ENSEMBLE_CONFIG = {
    'method': 'weighted_average',  # or 'stacking', 'blending'
    'weights': None,  # Auto-calculate based on validation performance
    'min_weight': 0.05,  # Minimum weight for any model
}

# ============================================================================
# TRAINING PIPELINE CONFIGURATION
# AUTO-OPTIMIZED BASED ON HARDWARE
# ============================================================================
TRAINING_CONFIG = {
    # Data preprocessing
    'use_extended_operational_data': False,
    
    # Bayesian optimization - AUTO-OPTIMIZED
    'n_trials': RECOMMENDED_N_TRIALS,
    
    # Cross-validation - AUTO-OPTIMIZED
    'cv_folds': RECOMMENDED_CV_FOLDS,
    'use_time_series_cv': False,
    
    # Model selection
    'use_autogluon': True,  # Enable AutoGluon AutoML
    'use_tabnet': True,     # Enable TabNet deep learning
    'time_limit_autogluon': 14400,
    
    # Feature engineering
    'use_polynomial_features': False,
    'use_physics_features': True,
    
    # Feature selection
    'use_feature_selection': True,
    
    # Ensemble
    'use_oof_ensemble': True,
    
    # Competition metrics
    'use_competition_metrics': True,
    'use_multi_objective': True,
    
    # Post-processing & Calibration
    'use_isotonic_calibration': True,
    'use_conformal_prediction': True,
    'use_tolerance_head': True,
    'use_monotonic_constraints': True,
    
    # Random seed for reproducibility
    'random_seed': RANDOM_SEED,
}

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================
FEATURE_CONFIG = {
    # Physics-informed features (Phase 2)
    'create_time_features': True,
    'create_temperature_features': True,
    'create_pressure_features': True,
    'create_flow_features': True,
    'create_efficiency_features': True,
    
    # Time-series features
    'lag_periods': [1, 2, 3, 5, 10],  # Lag windows for time-series
    'rolling_windows': [3, 5, 10, 20],  # Rolling statistics windows
    
    'variance_threshold': 0.01,
    'correlation_threshold': 0.95,
    'keep_ratio': 0.50,
    'use_shap': True,
}

# ============================================================================
# DATA AUGMENTATION CONFIGURATION (Phase 8)
# ============================================================================
AUGMENTATION_CONFIG = {
    'enabled': False,
    'use_noise': True,
    'use_jittering': True,
    'use_smote': False,
    
    # Noise augmentation
    'noise_augment_ratio': 0.25,
    'noise_level': 0.015, 
    
    # Jittering augmentation
    'jitter_augment_ratio': 0.20,
    'jitter_strength': 0.015,
}

# ============================================================================
# POST-PROCESSING & CALIBRATION (Phases 1-4 from Improvement Plan pt2)
# ============================================================================
CALIBRATION_CONFIG = {
    # Phase 1: Isotonic Calibration (post-hoc, near-free)
    'use_isotonic_calibration': True,
    'isotonic_increasing': True,  # Monotonic constraint
    'isotonic_out_of_bounds': 'clip',  # Handle extrapolation
    'isotonic_y_min': None,  # Auto-detect from training data
    'isotonic_y_max': None,  # Auto-detect from training data
    
    # Phase 2: Conformalized Quantile Regression
    'use_conformal_prediction': True,
    'conformal_alpha': 0.1,  # 90% prediction intervals
    'quantile_levels': [0.1, 0.5, 0.9],  # Lower, median, upper
    'use_quantile_median': True,  # Use median as point prediction
    
    # Phase 3: Tolerance Head (SVR with epsilon-insensitive loss)
    'use_tolerance_head': True,
    'tolerance_epsilon': 10.0,  # ±10 tolerance for competition metric
    'tolerance_blend_weight': 0.2,  # 20% SVR, 80% RMSE model (OOF-validated)
    'tolerance_C': 1.0,  # SVR regularization parameter
    'tolerance_kernel': 'rbf',  # RBF kernel for non-linearity
    
    # Phase 4: Monotonic Constraints (physics-informed)
    'use_monotonic_constraints': True,
    'monotonic_features': {
        # Format: 'feature_name': direction (1=increasing, -1=decreasing)
        # To be populated based on domain knowledge during training
        # Examples:
        # 'temperature_reactor': 1,  # Higher temp → Higher PCI
        # 'pressure_reactor': 1,     # Higher pressure → Higher H2
    },
    
    # Phase 5: Recency Weighting (sample_weight for time-series)
    'use_recency_weighting': True,
    'recency_decay': 0.01,  # Exponential decay rate (tune in CV)
    'recency_halflife_days': 30,  # Half-life for weight decay
    
    # Phase 6: Target Stabilization (Yeo-Johnson transform)
    'use_target_transform': True,
    'target_transform_method': 'yeo-johnson',  # PowerTransformer
    'target_standardize': False,  # Keep scale for interpretability
    
    # Validation and safety
    'calibration_fold_ratio': 0.2,  # Use 20% of train for calibration
    'validate_on_all_folds': True,  # Ensure consistent improvement across folds
    'min_improvement_threshold': 0.0,  # Only adopt if non-regressing
}

# ============================================================================
# TABNET CONFIGURATION (Phase 5)
# ============================================================================
TABNET_CONFIG = {
    'enabled': True,
    'advanced_mode': True,  # 2x capacity for physics features
    
    # Architecture (advanced mode)
    'n_d': 128,  # Width of decision prediction layer (doubled)
    'n_a': 128,  # Width of attention embedding (doubled)
    'n_steps': 6,  # Number of steps in architecture (increased)
    'gamma': 1.5,  # Coefficient for feature reusage
    
    # Regularization
    'lambda_sparse': 1e-4,
    'momentum': 0.3,
    
    # Training
    'max_epochs': 200,
    'patience': 30,
    'batch_size': 256,
    'virtual_batch_size': 128,
    'learning_rate': 0.02,
    
    # Optimizer
    'optimizer_fn': 'torch.optim.Adam',
    'scheduler_fn': 'torch.optim.lr_scheduler.ReduceLROnPlateau',
    'scheduler_params': {
        'mode': 'min',
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-5
    },
}

# ============================================================================
# AUTOGLUON CONFIGURATION
# ============================================================================
AUTOGLUON_CONFIG = {
    'enabled': True,
    'time_limit': 14400,
    'presets': 'best_quality',  # 'best_quality', 'high_quality', 'medium_quality'
    'num_bag_folds': 3,
    'num_bag_sets': 1,
    'num_stack_levels': 1,
    'verbosity': 2,
}

# ============================================================================
# COMPETITION METRICS CONFIGURATION (ANCAP 2025)
# ============================================================================
COMPETITION_CONFIG = {
    # Scoring weights
    'accuracy_weight': 0.30,  # 30% Model Accuracy
    'innovation_weight': 0.70,  # 70% Innovation & Implementation
    
    # Primary metrics
    'track_rmse_prom': True,  # (RMSE_PCI + RMSE_H2) / 2
    'track_within_10': True,  # Count predictions within ±10% RELATIVE error
    
    # Evaluation criteria (within ±10% RELATIVE tolerance)
    'tolerance_pct': 0.10,  # 10% relative tolerance (|error| / |true_value| <= 0.10)
    'tolerance_pci': 10,    # DEPRECATED: Was absolute, now using tolerance_pct
    'tolerance_h2': 10,     # DEPRECATED: Was absolute, now using tolerance_pct
    
    # Ranking: Average of 4 ranks (FCC/CCR × RMSE/±10%)
    'final_ranking_method': 'average_ranks',
    
    # Meta-ensemble weight validation
    'min_weight_threshold': 0.1,  # Warn if weight < 0.1 (tiny contribution)
    'expected_weight_min': 0.5,  # Expected healthy weight minimum
    'expected_weight_max': 1.0,  # Expected healthy weight maximum
}

# ============================================================================
# DATA QUALITY CONFIGURATION
# ============================================================================
DATA_QUALITY_CONFIG = {
    # Outlier detection
    'detect_outliers_iqr': True,
    'iqr_multiplier': 1.5,
    'detect_outliers_zscore': True,
    'zscore_threshold': 3,
    
    # Missing value handling
    'imputation_strategy': 'median',  # 'mean', 'median', 'most_frequent', 'constant'
    'imputation_constant': 0,
    
    # Feature checks
    'remove_constant_features': True,
    'constant_threshold': 0.99,
    
    # Adversarial validation
    'adversarial_validation': True,
    'adversarial_auc_threshold': 0.60,
}

# ============================================================================
# HARDWARE SUMMARY FUNCTION
# ============================================================================
def print_hardware_summary():
    """
    Print hardware detection summary and applied optimizations
    Usage: python -c "from config.model_config import print_hardware_summary; print_hardware_summary()"
    """
    if HARDWARE_AUTO_DETECTED:
        _HARDWARE_OPTIMIZER.print_summary()
        print("\n Hardware auto-detection ENABLED")
        print(f"   Configuration automatically optimized for your system")
    else:
        print("\n  Hardware auto-detection DISABLED")
        print(f"   Using default configuration values")
        print(f"   Install required packages: pip install torch gputil")
    
    print(f"\n Active Configuration:")
    print(f"   n_trials: {RECOMMENDED_N_TRIALS}")
    print(f"   timeout: {RECOMMENDED_TIMEOUT}s ({RECOMMENDED_TIMEOUT/3600:.1f}h)")
    print(f"   cv_folds: {RECOMMENDED_CV_FOLDS}")
    print(f"   n_jobs: {RECOMMENDED_N_JOBS}")
    print(f"   physical_cores: {NUM_PHYSICAL_CORES}")
    print(f"   gpu_available: {' Yes' if GPU_AVAILABLE else ' No'}")
    if GPU_AVAILABLE:
        print(f"   gpu_vram: {GPU_VRAM_GB:.1f} GB")


def get_hardware_info():
    """
    Get hardware information as dictionary
    Returns: dict with hardware specs and recommendations
    """
    if HARDWARE_AUTO_DETECTED:
        return {
            'auto_detected': True,
            'cpu_cores': NUM_PHYSICAL_CORES,
            'gpu_available': GPU_AVAILABLE,
            'gpu_vram_gb': GPU_VRAM_GB,
            'n_trials': RECOMMENDED_N_TRIALS,
            'timeout': RECOMMENDED_TIMEOUT,
            'cv_folds': RECOMMENDED_CV_FOLDS,
            'n_jobs': RECOMMENDED_N_JOBS,
            'optimizer': _HARDWARE_OPTIMIZER,
        }
    else:
        return {
            'auto_detected': False,
            'cpu_cores': NUM_PHYSICAL_CORES,
            'gpu_available': GPU_AVAILABLE,
            'gpu_vram_gb': GPU_VRAM_GB,
            'n_trials': RECOMMENDED_N_TRIALS,
            'timeout': RECOMMENDED_TIMEOUT,
            'cv_folds': RECOMMENDED_CV_FOLDS,
            'n_jobs': RECOMMENDED_N_JOBS,
        }
