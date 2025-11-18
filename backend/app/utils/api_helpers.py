"""
API Helper Functions for Backend Integration
Serialization, error handling, and utility functions
"""

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable
import pandas as pd
import numpy as np
from functools import wraps

from ..models.api_schemas import (
    ErrorResponse, ErrorDetail, 
    TrainingResponse, PredictionResponse,
    numpy_to_python, dataframe_to_features, features_to_dataframe
)


# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def serialize_predictions(predictions_pci: np.ndarray, 
                         predictions_h2: np.ndarray,
                         process_name: str,
                         features: Optional[pd.DataFrame] = None,
                         timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Serialize predictions to API-ready format
    
    Args:
        predictions_pci: PCI predictions
        predictions_h2: H2 predictions
        process_name: Process name
        features: Optional features DataFrame
        timestamps: Optional timestamps
    
    Returns:
        Dictionary ready for PredictionResponse
    """
    response = {
        'success': True,
        'process_name': process_name,
        'predictions_pci': predictions_pci.tolist() if isinstance(predictions_pci, np.ndarray) else predictions_pci,
        'predictions_h2': predictions_h2.tolist() if isinstance(predictions_h2, np.ndarray) else predictions_h2,
        'n_samples': len(predictions_pci),
        'statistics': {
            'pci': {
                'min': float(np.min(predictions_pci)),
                'max': float(np.max(predictions_pci)),
                'mean': float(np.mean(predictions_pci)),
                'std': float(np.std(predictions_pci))
            },
            'h2': {
                'min': float(np.min(predictions_h2)),
                'max': float(np.max(predictions_h2)),
                'mean': float(np.mean(predictions_h2)),
                'std': float(np.std(predictions_h2))
            }
        }
    }
    
    # Add features if provided
    if features is not None:
        response['features'], _ = dataframe_to_features(features)
    
    # Add timestamps if provided
    if timestamps is not None:
        response['timestamps'] = timestamps.astype(str).tolist()
    
    return response


def serialize_training_results(results: Dict[str, Any],
                               output_dir: str) -> Dict[str, Any]:
    """
    Serialize training results to API-ready format
    
    Args:
        results: Training results from MLTrainer
        output_dir: Directory where models were saved
    
    Returns:
        Dictionary ready for TrainingResponse
    """
    return {
        'success': True,
        'process_name': results['process_name'],
        'metrics_pci': numpy_to_python(results['metrics_pci']),
        'metrics_h2': numpy_to_python(results['metrics_h2']),
        'competition_score': numpy_to_python(results['competition_score']),
        'training_time': float(results['training_time']),
        'n_samples_train': int(results['n_samples_train']),
        'n_samples_test': int(results['n_samples_test']),
        'model_path': str(output_dir),
        'message': f"Training completed successfully for {results['process_name']}"
    }


def serialize_dataframe_predictions(df: pd.DataFrame,
                                   process_name: str,
                                   pci_col: str = 'PCI_pred',
                                   h2_col: str = 'H2_pred') -> Dict[str, Any]:
    """
    Serialize DataFrame with predictions to API format
    
    Args:
        df: DataFrame with predictions
        process_name: Process name
        pci_col: PCI prediction column name
        h2_col: H2 prediction column name
    
    Returns:
        Dictionary ready for PredictionResponse
    """
    timestamps = None
    if 'sampled_date' in df.columns:
        timestamps = df['sampled_date']
    elif 'timestamp' in df.columns:
        timestamps = df['timestamp']
    
    return serialize_predictions(
        predictions_pci=df[pci_col].values,
        predictions_h2=df[h2_col].values,
        process_name=process_name,
        timestamps=timestamps
    )


# ============================================================================
# ERROR HANDLING
# ============================================================================

def create_error_response(error: Exception,
                         error_code: str = 'INTERNAL_ERROR',
                         include_traceback: bool = False) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error: Exception that occurred
        error_code: Error code
        include_traceback: Include full traceback (for debugging)
    
    Returns:
        Dictionary ready for ErrorResponse
    """
    error_detail = {
        'error_code': error_code,
        'message': str(error),
        'timestamp': datetime.now().isoformat(),
        'details': {
            'type': type(error).__name__
        }
    }
    
    if include_traceback:
        error_detail['traceback'] = traceback.format_exc()
    
    return {
        'success': False,
        'error': error_detail
    }


def handle_api_errors(error_code: str = 'INTERNAL_ERROR',
                     include_traceback: bool = False):
    """
    Decorator for handling API errors
    
    Args:
        error_code: Default error code
        include_traceback: Include traceback in response
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                return create_error_response(e, 'VALIDATION_ERROR', include_traceback)
            except FileNotFoundError as e:
                return create_error_response(e, 'FILE_NOT_FOUND', include_traceback)
            except KeyError as e:
                return create_error_response(e, 'KEY_ERROR', include_traceback)
            except Exception as e:
                return create_error_response(e, error_code, include_traceback)
        return wrapper
    return decorator


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_process_name(process_name: str) -> str:
    """
    Validate and normalize process name
    
    Args:
        process_name: Process name to validate
    
    Returns:
        Normalized process name
    
    Raises:
        ValueError: If process name is invalid
    """
    process_name = process_name.upper().strip()
    if process_name not in ['FCC', 'CCR']:
        raise ValueError(f"Invalid process name: {process_name}. Must be 'FCC' or 'CCR'")
    return process_name


def validate_model_exists(model_dir: str, process_name: str) -> Path:
    """
    Validate that model directory and files exist
    
    Args:
        model_dir: Model directory
        process_name: Process name
    
    Returns:
        Path to model directory
    
    Raises:
        FileNotFoundError: If model files not found
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Check for ensemble files
    pci_ensemble = model_path / f'{process_name}_ensemble_pci.joblib'
    h2_ensemble = model_path / f'{process_name}_ensemble_h2.joblib'
    
    if not pci_ensemble.exists() or not h2_ensemble.exists():
        raise FileNotFoundError(
            f"Model files not found for {process_name} in {model_dir}. "
            f"Please train models first."
        )
    
    return model_path


def validate_features_shape(features: List[List[float]], 
                           expected_cols: Optional[int] = None) -> None:
    """
    Validate feature matrix shape
    
    Args:
        features: Feature matrix as list of lists
        expected_cols: Expected number of columns (optional)
    
    Raises:
        ValueError: If shape is invalid
    """
    if len(features) == 0:
        raise ValueError("Feature matrix is empty")
    
    # Check all rows have same length
    first_len = len(features[0])
    if not all(len(row) == first_len for row in features):
        raise ValueError("All feature rows must have the same length")
    
    # Check expected columns
    if expected_cols is not None and first_len != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} features, got {first_len}"
        )


# ============================================================================
# DATA CONVERSION HELPERS
# ============================================================================

def request_to_dataframe(features: List[List[float]],
                        feature_names: Optional[List[str]] = None,
                        targets_pci: Optional[List[float]] = None,
                        targets_h2: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Convert API request data to DataFrame
    
    Args:
        features: Feature matrix
        feature_names: Feature column names
        targets_pci: PCI target values (optional)
        targets_h2: H2 target values (optional)
    
    Returns:
        DataFrame with features and targets
    """
    # Create feature DataFrame
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(features[0]))]
    
    df = pd.DataFrame(features, columns=feature_names)
    
    # Add targets if provided
    if targets_pci is not None:
        df['PCI'] = targets_pci
    if targets_h2 is not None:
        df['H2'] = targets_h2
    
    return df


def csv_to_api_format(csv_path: str,
                     feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convert CSV file to API request format
    
    Args:
        csv_path: Path to CSV file
        feature_cols: Feature column names (if None, auto-detect)
    
    Returns:
        Dictionary with features and metadata
    """
    df = pd.read_csv(csv_path)
    
    # Auto-detect feature columns
    if feature_cols is None:
        exclude_cols = ['PCI', 'H2', 'sampled_date', 'timestamp', 'id', 'ID']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    features, feature_names = dataframe_to_features(df, feature_cols)
    
    result = {
        'features': features,
        'feature_names': feature_names,
        'n_samples': len(df)
    }
    
    # Add targets if available
    if 'PCI' in df.columns:
        result['targets_pci'] = df['PCI'].tolist()
    if 'H2' in df.columns:
        result['targets_h2'] = df['H2'].tolist()
    
    return result


# ============================================================================
# MODEL MANAGEMENT HELPERS
# ============================================================================

def list_available_models(model_dir: str = 'models') -> List[Dict[str, Any]]:
    """
    List all available trained models
    
    Args:
        model_dir: Model directory
    
    Returns:
        List of model information dictionaries
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        return []
    
    models = []
    
    # Look for ensemble files
    for ensemble_file in model_path.glob('*_ensemble_pci.joblib'):
        process_name = ensemble_file.stem.replace('_ensemble_pci', '')
        
        # Get all related files
        model_files = [
            str(f.relative_to(model_path))
            for f in model_path.glob(f'{process_name}*')
        ]
        
        # Get training history
        history_file = model_path / f'{process_name}_training_history.json'
        training_history = None
        if history_file.exists():
            import json
            with open(history_file, 'r') as f:
                training_history = json.load(f)
        
        # Calculate total size
        total_size = sum(
            f.stat().st_size for f in model_path.glob(f'{process_name}*')
        ) / (1024 * 1024)  # Convert to MB
        
        models.append({
            'process_name': process_name,
            'model_path': str(model_path),
            'model_files': model_files,
            'training_history': training_history,
            'created_at': datetime.fromtimestamp(
                ensemble_file.stat().st_mtime
            ).isoformat(),
            'file_size_mb': round(total_size, 2)
        })
    
    return models


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of ML dependencies
    
    Returns:
        Dictionary of dependency availability
    """
    dependencies = {}
    
    try:
        import xgboost
        dependencies['xgboost'] = True
    except ImportError:
        dependencies['xgboost'] = False
    
    try:
        import lightgbm
        dependencies['lightgbm'] = True
    except ImportError:
        dependencies['lightgbm'] = False
    
    try:
        import catboost
        dependencies['catboost'] = True
    except ImportError:
        dependencies['catboost'] = False
    
    try:
        import autogluon
        dependencies['autogluon'] = True
    except ImportError:
        dependencies['autogluon'] = False
    
    try:
        import optuna
        dependencies['optuna'] = True
    except ImportError:
        dependencies['optuna'] = False
    
    dependencies['sklearn'] = True  # Always available
    dependencies['pandas'] = True  # Always available
    dependencies['numpy'] = True   # Always available
    
    return dependencies


if __name__ == '__main__':
    print("API Helper Functions")
    print("\nAvailable functions:")
    print("  Serialization: serialize_predictions, serialize_training_results")
    print("  Error Handling: create_error_response, handle_api_errors")
    print("  Validation: validate_process_name, validate_model_exists")
    print("  Conversion: request_to_dataframe, csv_to_api_format")
    print("  Management: list_available_models, check_dependencies")
